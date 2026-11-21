"""Feature engineering - turns the raw daily table into model inputs.

LEAKAGE RULE: every feature for a row at date t may use data from dates
<= t ONLY. All rolling windows below look backwards. tests/test_features.py
proves this by changing the future and checking that past features do not move.

Product ID is deliberately NOT a feature: with ~60 series it would let a tree
memorise individual products instead of learning general inventory risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import utilities as U

EPS = 1e-6
_NO_STOCKOUT_CAP = 60  # days_since_last_stockout when none seen (capped)

NUMERIC_FEATURES = [
    "current_stock",
    "avg_daily_demand_7d",
    "avg_daily_demand_28d",
    "demand_cv_28d",
    "demand_trend_ratio",
    "days_of_cover",
    "lead_time_days",
    "cover_minus_lead_time",
    "reorder_point",
    "stock_to_reorder_ratio",
    "open_order_qty",
    "open_order_cover_days",
    "order_pending",
    "order_age_days",
    "stockouts_last_28d",
    "days_since_last_stockout",
    "day_of_week",
    "month",
]
CATEGORICAL_FEATURES = [U.CATEGORY, U.WAREHOUSE]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Human-readable names, used by the explanation module and UI.
FEATURE_LABELS = {
    "current_stock": "Current stock (units)",
    "avg_daily_demand_7d": "Average daily sales, last 7 days",
    "avg_daily_demand_28d": "Average daily sales, last 28 days",
    "demand_cv_28d": "Demand variability (CV, 28 days)",
    "demand_trend_ratio": "Recent demand vs. 28-day demand",
    "days_of_cover": "Days of inventory remaining",
    "lead_time_days": "Replenishment lead time (days)",
    "cover_minus_lead_time": "Days of cover minus lead time",
    "reorder_point": "Reorder point (units)",
    "stock_to_reorder_ratio": "Stock / reorder point",
    "open_order_qty": "Units on order",
    "open_order_cover_days": "Days of demand covered by open order",
    "order_pending": "Replenishment order pending",
    "order_age_days": "Days since open order was placed",
    "stockouts_last_28d": "Stockout days in last 28 days",
    "days_since_last_stockout": "Days since last stockout",
    "day_of_week": "Day of week (0=Mon)",
    "month": "Month",
    U.CATEGORY: "Product category",
    U.WAREHOUSE: "Warehouse",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the table with engineered feature columns appended.

    Input must use canonical column names and pass validation (daily rows
    without gaps). `order_age_days` and `warehouse` are optional in the raw
    data: if absent they are set to 0 / "All" (a neutral label, not a
    fabricated measurement; the UI warns about it).
    """
    out = df.copy()
    out[U.DATE] = pd.to_datetime(out[U.DATE])
    if U.WAREHOUSE not in out.columns:
        out[U.WAREHOUSE] = "All"
    if "order_age_days" not in out.columns:
        out["order_age_days"] = 0
    keys = U.SERIES_KEYS
    out = out.sort_values(keys + [U.DATE]).reset_index(drop=True)
    g = out.groupby(keys, sort=False)

    # --- rolling demand statistics (window ends at t, includes day t only)
    sold = out[U.UNITS_SOLD].astype(float)
    r7 = sold.groupby([out[k] for k in keys]).rolling(U.SHORT_WINDOW, min_periods=1)
    r28 = sold.groupby([out[k] for k in keys]).rolling(U.LONG_WINDOW, min_periods=U.SHORT_WINDOW)
    out["avg_daily_demand_7d"] = r7.mean().reset_index(level=list(range(len(keys))), drop=True).sort_index()
    out["avg_daily_demand_28d"] = r28.mean().reset_index(level=list(range(len(keys))), drop=True).sort_index()
    std28 = r28.std().reset_index(level=list(range(len(keys))), drop=True).sort_index()
    out["demand_cv_28d"] = std28 / (out["avg_daily_demand_28d"] + EPS)
    out["demand_trend_ratio"] = out["avg_daily_demand_7d"] / (out["avg_daily_demand_28d"] + EPS)

    # --- stock coverage
    out["current_stock"] = out[U.CLOSING_STOCK].astype(float)
    out["days_of_cover"] = out["current_stock"] / (out["avg_daily_demand_28d"] + EPS)
    out["days_of_cover"] = out["days_of_cover"].clip(upper=365)
    out["cover_minus_lead_time"] = out["days_of_cover"] - out[U.LEAD_TIME]
    out["stock_to_reorder_ratio"] = out["current_stock"] / (out[U.REORDER_POINT] + EPS)
    out["stock_to_reorder_ratio"] = out["stock_to_reorder_ratio"].clip(upper=50)

    # --- replenishment
    out["open_order_qty"] = out[U.OPEN_ORDER_QTY].astype(float)
    out["open_order_cover_days"] = (out["open_order_qty"] / (out["avg_daily_demand_28d"] + EPS)).clip(upper=365)
    out["order_pending"] = (out["open_order_qty"] > 0).astype(int)
    out["order_age_days"] = out["order_age_days"].astype(float)

    # --- stockout history (today's flag is known at end of day, so it is allowed)
    flag = out[U.STOCKOUT_FLAG].astype(float)
    out["stockouts_last_28d"] = (
        flag.groupby([out[k] for k in keys]).rolling(U.LONG_WINDOW, min_periods=1).sum()
        .reset_index(level=list(range(len(keys))), drop=True).sort_index()
    )
    # days since last stockout: cumulative counter that resets at each stockout
    last_so_date = out[U.DATE].where(flag == 1)
    last_so_date = last_so_date.groupby([out[k] for k in keys]).ffill()
    since = (out[U.DATE] - last_so_date).dt.days
    out["days_since_last_stockout"] = since.fillna(_NO_STOCKOUT_CAP).clip(upper=_NO_STOCKOUT_CAP)

    # --- calendar
    out["day_of_week"] = out[U.DATE].dt.dayofweek
    out["month"] = out[U.DATE].dt.month

    # --- history length: used to drop rows whose rolling windows are unreliable
    out["history_days"] = g.cumcount() + 1
    return out


def usable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with enough history and complete feature values."""
    mask = (df["history_days"] >= U.MIN_HISTORY_DAYS) & df[FEATURE_COLUMNS].notna().all(axis=1)
    return df[mask].copy()
