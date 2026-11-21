"""Running a trained model on inventory data."""
from __future__ import annotations

import pandas as pd

from src import utilities as U
from src.feature_engineering import FEATURE_COLUMNS, build_features, usable_rows

HIGH, LOW = "High Risk", "Low Risk"


def prepare_for_prediction(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build features for validated canonical-column data.

    Returns (usable_rows, info). Rows lacking enough history or complete inputs
    are EXCLUDED (never filled in); `info` says how many, so the UI can report it.
    """
    feats = build_features(df_raw)
    usable = usable_rows(feats)
    info = {"rows_in": int(len(df_raw)), "rows_usable": int(len(usable)),
            "rows_excluded": int(len(df_raw) - len(usable))}
    return usable, info


def predict(bundle: dict, feats: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Add risk columns to a feature table.

    Adds: risk_probability (tree-leaf risk score, 0-1), risk_class (1/0),
    risk_label ('High Risk'/'Low Risk') and risk_rank (1 = highest score).
    """
    missing = [c for c in bundle["feature_columns"] if c not in feats.columns]
    if missing:
        raise ValueError(f"Cannot predict: feature columns missing: {missing}. "
                         "Run the data through build_features() first.")
    if feats[bundle["feature_columns"]].isna().any().any():
        raise ValueError("Cannot predict: some feature values are missing. "
                         "Pass the output of prepare_for_prediction(), which excludes incomplete rows.")
    thr = bundle["config"]["threshold"] if threshold is None else threshold
    if not 0 < thr < 1:
        raise ValueError("threshold must be between 0 and 1")

    out = feats.copy()
    if out.empty:
        for c, v in (("risk_probability", 0.0), ("risk_class", 0), ("risk_label", LOW), ("risk_rank", 0)):
            out[c] = v
        return out
    out["risk_probability"] = bundle["pipeline"].predict_proba(out[bundle["feature_columns"]])[:, 1]
    out["risk_class"] = (out["risk_probability"] >= thr).astype(int)
    out["risk_label"] = out["risk_class"].map({1: HIGH, 0: LOW})
    out["risk_rank"] = out["risk_probability"].rank(method="first", ascending=False).astype(int)
    return out


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """The most recent row of every product-warehouse series (the 'today' view)."""
    idx = df.groupby(U.SERIES_KEYS)[U.DATE].idxmax()
    return df.loc[idx].reset_index(drop=True)


def predict_from_raw(bundle: dict, df_raw: pd.DataFrame, latest_only: bool = False,
                     threshold: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Raw validated table -> predictions (+ info about excluded rows)."""
    usable, info = prepare_for_prediction(df_raw)
    if latest_only:
        usable = latest_snapshot(usable)
    return predict(bundle, usable, threshold), info


def to_prediction_table(pred: pd.DataFrame) -> pd.DataFrame:
    """Compact table in the layout of the Predictions page (Product, Category, Stock, ...)."""
    cols = {
        U.PRODUCT_ID: "Product ID",
        U.PRODUCT_NAME: "Product",
        U.CATEGORY: "Category",
        U.WAREHOUSE: "Warehouse",
        "current_stock": "Stock",
        "avg_daily_demand_28d": "Demand (units/day)",
        "days_of_cover": "Days Remaining",
        U.LEAD_TIME: "Lead Time (days)",
        "risk_label": "Risk",
        "risk_probability": "Risk Probability",
    }
    present = {k: v for k, v in cols.items() if k in pred.columns}
    return (pred[list(present)].rename(columns=present)
            .sort_values("Risk Probability", ascending=False).reset_index(drop=True))
