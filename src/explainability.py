"""Explaining the decision tree: feature importance, tree plots and per-row decision paths.

Every sentence is produced by code from (a) the thresholds the row actually crossed
in the fitted tree and (b) the row's actual values. No generative model is involved.

Feature importance and decision paths describe how the TREE behaves on this data.
They are NOT evidence that a feature causes stockouts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import export_text, plot_tree

from src.feature_engineering import _NO_STOCKOUT_CAP as _STOCKOUT_CAP
from src.feature_engineering import CATEGORICAL_FEATURES, FEATURE_LABELS
from src.preprocessing import original_feature_of, transformed_feature_names

LIMITATIONS_TEXT = (
    "Feature importance and decision paths show how this tree separates the training data. "
    "They do not prove that a feature causes stockouts, and a different training period could "
    "produce a different tree. The risk score is a leaf proportion, not a calibrated probability."
)


def _tree(bundle):
    return bundle["pipeline"].named_steps["tree"]


def _label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


# ---------------------------------------------------------- feature importance
def feature_importance(bundle: dict, grouped: bool = True) -> pd.DataFrame:
    """Impurity-based importance. `grouped=True` sums one-hot columns back to category/warehouse."""
    names = transformed_feature_names(bundle["pipeline"])
    imp = pd.DataFrame({"feature": names, "importance": _tree(bundle).feature_importances_})
    if grouped:
        imp["feature"] = imp["feature"].map(original_feature_of)
        imp = imp.groupby("feature", as_index=False)["importance"].sum()
    imp["label"] = imp["feature"].map(_label)
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------ tree rendering
def render_tree(bundle: dict, max_depth: int = 3, figsize=(22, 10)):
    """Matplotlib Figure of the top `max_depth` levels of the tree."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # headless-safe renderer
    from matplotlib.figure import Figure
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)          # plot_tree needs a canvas that can measure text
    ax = fig.subplots()
    plot_tree(_tree(bundle), feature_names=transformed_feature_names(bundle["pipeline"]),
              class_names=["Low Risk", "High Risk"], filled=True, rounded=True, impurity=False,
              proportion=False, max_depth=max_depth, fontsize=9, ax=ax)
    fig.tight_layout()
    return fig


def tree_text(bundle: dict, max_depth: int = 4) -> str:
    """Plain-text rules of the tree (readable in a code block or report)."""
    return export_text(_tree(bundle), feature_names=transformed_feature_names(bundle["pipeline"]),
                       max_depth=max_depth)


def tree_summary(bundle: dict) -> dict:
    t = _tree(bundle)
    return {"depth": int(t.get_depth()), "n_leaves": int(t.get_n_leaves()), "n_nodes": int(t.tree_.node_count)}


# --------------------------------------------------------------- decision path
def _transform_row(bundle: dict, row: pd.DataFrame) -> np.ndarray:
    """Preprocess rows exactly as the tree sees them.

    scikit-learn trees compare float32 values internally; casting here keeps our
    reported branches identical to the branches the model actually took.
    """
    X = bundle["pipeline"].named_steps["prep"].transform(row[bundle["feature_columns"]])
    return np.asarray(X, dtype=np.float32)


def decision_path(bundle: dict, row: pd.DataFrame) -> dict:
    """Walk one product-day through the tree.

    `row` is a one-row DataFrame containing the model feature columns.
    Returns the ordered list of rules followed and the leaf reached.
    """
    if len(row) != 1:
        raise ValueError("decision_path expects exactly one row")
    names = transformed_feature_names(bundle["pipeline"])
    tree = _tree(bundle)
    X = _transform_row(bundle, row)
    node_ids = tree.decision_path(X).indices            # nodes visited, root -> leaf
    leaf = int(tree.apply(X)[0])

    steps = []
    for node in node_ids:
        if node == leaf:
            continue
        f_idx = tree.tree_.feature[node]
        thr = float(tree.tree_.threshold[node])
        value = float(X[0, f_idx])
        went_left = value <= thr
        steps.append({
            "node": int(node), "transformed_feature": names[f_idx],
            "feature": original_feature_of(names[f_idx]), "threshold": thr, "value": value,
            "operator": "<=" if went_left else ">",
            "text": _step_text(names[f_idx], thr, value, went_left),
        })
    proba = tree.predict_proba(X)[0]
    return {
        "steps": steps, "leaf_node": leaf, "leaf_samples": int(tree.tree_.n_node_samples[leaf]),
        "risk_probability": float(proba[1]), "n_steps": len(steps),
    }


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"


def _value_str(tname: str, value: float) -> str:
    """Display a feature value; capped features show their cap honestly (e.g. '60+')."""
    if tname == "days_since_last_stockout" and value >= _STOCKOUT_CAP:
        return f"{_STOCKOUT_CAP}+"   # capped: no stockout in the last 60+ days
    return _fmt(value)


def _step_text(tname: str, thr: float, value: float, went_left: bool) -> str:
    """One rule in words, e.g. 'Days of inventory remaining is 3.10 (rule: at most 4.20)'."""
    orig = original_feature_of(tname)
    if orig in CATEGORICAL_FEATURES:                     # one-hot column: threshold is 0.5
        level = tname[len(orig) + 1:]
        return f"{_label(orig)} is {level}" if value > thr else f"{_label(orig)} is not {level}"
    rule = f"at most {_fmt(thr)}" if went_left else f"above {_fmt(thr)}"
    return f"{_label(tname)} is {_value_str(tname, value)} (rule: {rule})"


def _merge_numeric_bounds(steps: list[dict]) -> list[str]:
    """Collapse repeated splits on one feature into a single tightest condition."""
    order, info = [], {}
    for s in steps:
        key = s["transformed_feature"]
        if key not in info:
            order.append(key)
            info[key] = {"upper": np.inf, "lower": -np.inf, "step": s}
        if s["operator"] == "<=":
            info[key]["upper"] = min(info[key]["upper"], s["threshold"])
        else:
            info[key]["lower"] = max(info[key]["lower"], s["threshold"])
    phrases = []
    for key in order:
        d, s = info[key], info[key]["step"]
        if original_feature_of(key) in CATEGORICAL_FEATURES:
            phrases.append(s["text"])
            continue
        lo, hi, label, val = d["lower"], d["upper"], _label(key), _value_str(key, s["value"])
        if np.isfinite(lo) and np.isfinite(hi):
            cond = f"between {_fmt(lo)} and {_fmt(hi)}"
        elif np.isfinite(hi):
            cond = f"at most {_fmt(hi)}"
        else:
            cond = f"above {_fmt(lo)}"
        phrases.append(f"{label} is {val} ({cond})")
    return phrases


def explain_row(bundle: dict, row: pd.DataFrame, threshold: float | None = None) -> dict:
    """Full explanation for one product-day: prediction, path, and a sentence built from them."""
    thr = bundle["config"]["threshold"] if threshold is None else threshold
    path = decision_path(bundle, row)
    score = path["risk_probability"]
    label = "High Risk" if score >= thr else "Low Risk"
    conditions = _merge_numeric_bounds(path["steps"])
    if conditions:
        body = "the decision path found: " + "; ".join(conditions) + "."
    else:
        body = "the tree made no splits (it predicts the same class for every product)."
    summary = (f"Predicted {label} (risk score {score:.0%}) because {body} "
               f"This pattern matched {path['leaf_samples']:,} training rows.")
    return {
        "predicted_label": label, "risk_probability": score, "threshold": thr,
        "input_values": {f: row.iloc[0][f] for f in bundle["feature_columns"]},
        "steps": path["steps"], "conditions": conditions, "summary": summary,
        "leaf_node": path["leaf_node"], "leaf_samples": path["leaf_samples"],
        "limitations": LIMITATIONS_TEXT,
    }


def path_text_for_rows(bundle: dict, feats: pd.DataFrame) -> pd.Series:
    """Compact one-line decision path for MANY rows (e.g. the High-Risk page table).

    Uses a single sparse decision_path call, so it is fast for thousands of rows.
    """
    if feats.empty:
        return pd.Series([], dtype=object, index=feats.index)
    names = transformed_feature_names(bundle["pipeline"])
    tree = _tree(bundle)
    X = _transform_row(bundle, feats)
    indicator = tree.decision_path(X)
    leaves = tree.apply(X)
    out = []
    for i in range(X.shape[0]):
        nodes = indicator.indices[indicator.indptr[i]:indicator.indptr[i + 1]]
        parts = []
        for node in nodes:
            if node == leaves[i]:
                continue
            f = tree.tree_.feature[node]
            thr = tree.tree_.threshold[node]
            left = X[i, f] <= thr
            if original_feature_of(names[f]) in CATEGORICAL_FEATURES:
                lvl = names[f][len(original_feature_of(names[f])) + 1:]
                parts.append(f"{original_feature_of(names[f])}{'=' if not left else '≠'}{lvl}")
            else:
                parts.append(f"{names[f]} {'<=' if left else '>'} {_fmt(thr)}")
        out.append(" → ".join(parts))
    return pd.Series(out, index=feats.index)


def step_is_satisfied(step: dict) -> bool:
    """True if the recorded actual value really satisfies the recorded rule (used in tests/audits)."""
    return step["value"] <= step["threshold"] if step["operator"] == "<=" else step["value"] > step["threshold"]
