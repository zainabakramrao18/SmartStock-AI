"""Target-label construction.

Definition
----------
For a row at date t of one series (product x warehouse):

    stockout_within_horizon = 1  if stockout_flag == 1 on ANY day t+1 ... t+H
                              0  if stockout_flag == 0 on ALL of those days
                              NaN if the future window is not fully observed
                                   and no stockout has been seen yet

Why this is not leakage
-----------------------
The label is built from FUTURE rows only (t+1 onwards). It is used only as
the answer ("y") during training and is never fed in as a feature. Today's
stockout flag (day t) is a legitimate feature because it is already known
at the end of day t.

Rows near the end of a series have an incomplete future window. They get
NaN (unknown) and must be excluded from training/testing - we never
pretend "no stockout was recorded" means "no stockout happened".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import utilities as U


def add_target(df: pd.DataFrame, horizon: int = U.HORIZON_DAYS) -> pd.DataFrame:
    """Return a copy of `df` with the `stockout_within_horizon` column added.

    The table must contain date, product_id, warehouse and stockout_flag.
    Dates inside a series do not have to be contiguous: a future day only
    counts if it is exactly k calendar days after t, so gaps yield NaN
    rather than a wrong label.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1 day")
    out = df.copy()
    out[U.DATE] = pd.to_datetime(out[U.DATE])
    out = out.sort_values(U.SERIES_KEYS + [U.DATE]).reset_index(drop=True)

    grp = out.groupby(U.SERIES_KEYS, sort=False)
    any_stockout = np.zeros(len(out), dtype=bool)
    all_observed = np.ones(len(out), dtype=bool)

    for k in range(1, horizon + 1):
        future_flag = grp[U.STOCKOUT_FLAG].shift(-k)
        future_date = grp[U.DATE].shift(-k)
        # A future day only counts as observed if it exists exactly k days later
        # AND its stockout flag is not missing (a blank flag is "unknown", not "0").
        observed = ((future_date - out[U.DATE]).dt.days == k) & future_flag.notna()
        all_observed &= observed.to_numpy()
        any_stockout |= (observed & (future_flag == 1)).to_numpy()

    label = np.where(any_stockout, 1.0, np.where(all_observed, 0.0, np.nan))
    out[U.TARGET] = label
    return out


def labelled_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose label is known (drops the unobservable tail)."""
    return df[df[U.TARGET].notna()].copy()
