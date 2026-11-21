"""Model evaluation: every number is computed from real predictions on the hold-out set.

Recall for the stockout class (label 1) is the headline metric, because a missed
stockout (false negative) is the costly error. Accuracy is reported but, with
only ~8-16 % positives, a model that always says "Low Risk" would already score
84-92 % accuracy while catching nothing - hence the baselines below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src import utilities as U


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Headline metrics + confusion-matrix counts for the stockout class (1)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "roc_auc": None, "pr_auc": None,
    }
    # AUCs are undefined when only one class is present in y_true
    if y_proba is not None and len(np.unique(y_true)) == 2:
        m["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        m["pr_auc"] = float(average_precision_score(y_true, y_proba))
    return m


def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=[0, 1], target_names=["Low Risk (0)", "High Risk (1)"],
                                output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T


def roc_curve_data(y_true, y_proba) -> pd.DataFrame | None:
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})


def pr_curve_data(y_true, y_proba) -> pd.DataFrame | None:
    if len(np.unique(y_true)) < 2:
        return None
    p, r, thr = precision_recall_curve(y_true, y_proba)
    return pd.DataFrame({"precision": p[:-1], "recall": r[:-1], "threshold": thr})


def threshold_table(y_true, y_proba, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> pd.DataFrame:
    """Precision / recall / flagged share at several decision thresholds.

    Lets a manager see the trade-off: a lower threshold catches more stockouts
    (higher recall) but flags more products that would not have stocked out.
    """
    y_true = np.asarray(y_true).astype(int)
    rows = []
    for t in thresholds:
        pred = (np.asarray(y_proba) >= t).astype(int)
        rows.append({
            "threshold": t,
            "recall": recall_score(y_true, pred, zero_division=0),
            "precision": precision_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "flagged_share": float(pred.mean()),
            "missed_stockouts": int(((pred == 0) & (y_true == 1)).sum()),
            "false_alarms": int(((pred == 1) & (y_true == 0)).sum()),
        })
    return pd.DataFrame(rows)


def calibration_table(y_true, y_proba, n_bins: int = 10) -> pd.DataFrame:
    """Predicted score vs. observed stockout rate per score bin.

    A decision tree's leaf proportions (especially with class weights) are NOT
    guaranteed to be true probabilities. This table measures how far off they are.
    """
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(y_proba)})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    out = (df.groupby("bin", observed=True)
             .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
             .reset_index())
    out["bin"] = out["bin"].astype(str)
    return out


def baseline_metrics(test_df: pd.DataFrame) -> dict[str, dict]:
    """Reference points the tree must beat to be worth using.

    * all_low_risk:   never raise an alarm (high accuracy, zero recall)
    * all_high_risk:  flag everything (perfect recall, useless precision)
    * cover_lt_lead_time: simple manager rule - flag when days of cover is shorter
      than the supplier lead time.
    """
    y = test_df[U.TARGET].astype(int).to_numpy()
    n = len(y)
    return {
        "all_low_risk": compute_metrics(y, np.zeros(n)),
        "all_high_risk": compute_metrics(y, np.ones(n)),
        "cover_lt_lead_time": compute_metrics(y, (test_df["cover_minus_lead_time"] < 0).astype(int).to_numpy()),
    }


def evaluate(pipeline, test_df: pd.DataFrame, feature_columns: list[str], threshold: float = 0.5) -> dict:
    """Full evaluation bundle on labelled hold-out rows."""
    y = test_df[U.TARGET].astype(int).to_numpy()
    proba = pipeline.predict_proba(test_df[feature_columns])[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "metrics": compute_metrics(y, pred, proba),
        "baselines": baseline_metrics(test_df),
        "classification_report": classification_report_df(y, pred),
        "roc_curve": roc_curve_data(y, proba),
        "pr_curve": pr_curve_data(y, proba),
        "threshold_table": threshold_table(y, proba),
        "calibration": calibration_table(y, proba),
    }


def interpret_metrics(m: dict, baselines: dict | None = None) -> str:
    """Plain-language reading of the metrics, assembled from the actual numbers."""
    pct = lambda v: f"{v * 100:.1f}%"
    lines = [
        f"On the hold-out period the model correctly flagged {m['true_positives']:,} of "
        f"{m['n_positive']:,} products that went on to stock out (recall {pct(m['recall'])}) and "
        f"missed {m['false_negatives']:,}.",
        f"Of everything it flagged as High Risk, {pct(m['precision'])} did stock out "
        f"({m['false_positives']:,} false alarms), so about {m['false_positives'] / max(m['true_positives'], 1):.1f} "
        f"unnecessary checks per real stockout caught.",
        f"Accuracy is {pct(m['accuracy'])}, but only {pct(m['positive_rate'])} of rows are actual stockouts, "
        f"so accuracy alone is misleading: always answering 'Low Risk' would score "
        f"{pct(1 - m['positive_rate'])} while catching nothing.",
    ]
    if m.get("roc_auc") is not None:
        lines.append(f"ROC-AUC is {m['roc_auc']:.2f} and PR-AUC is {m['pr_auc']:.2f} "
                     f"(PR-AUC of a random scorer would be about {m['positive_rate']:.2f}).")
    if baselines and "cover_lt_lead_time" in baselines:
        b = baselines["cover_lt_lead_time"]
        lines.append(f"For comparison, the simple rule 'days of cover < lead time' reaches recall "
                     f"{pct(b['recall'])} and precision {pct(b['precision'])}.")
    lines.append("These figures come from synthetic data and say nothing about real-world performance.")
    return " ".join(lines)
