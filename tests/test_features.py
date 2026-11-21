"""Tests: target generation, feature engineering and LEAKAGE guarantees."""
import numpy as np
import pandas as pd

from src import utilities as U
from src.feature_engineering import FEATURE_COLUMNS, build_features, usable_rows
from src.target_generation import add_target, labelled_rows


def _toy(flags, start="2025-01-01"):
    n = len(flags)
    return pd.DataFrame({U.DATE: pd.date_range(start, periods=n), U.PRODUCT_ID: "P1",
                         U.WAREHOUSE: "W1", U.STOCKOUT_FLAG: flags})


# ------------------------------------------------------------------ target
def test_target_looks_only_at_future_days():
    flags = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    out = add_target(_toy(flags), horizon=3)
    t = out[U.TARGET].tolist()
    assert t[0] == 1 and t[1] == 1      # stockout on day 3 lies within next 3 days
    assert t[2] == 0                    # today's own stockout does NOT count
    assert t[3] == 0
    assert np.isnan(t[-1]) and np.isnan(t[-2]) and np.isnan(t[-3])  # unobservable tail
    assert t[8] == 0                    # fully observed, no stockout


def test_target_positive_even_if_window_incomplete():
    out = add_target(_toy([0, 0, 0, 0, 1]), horizon=3)
    assert out[U.TARGET].iloc[2] == 1   # stockout seen at t+2 although window is cut short
    assert out[U.TARGET].iloc[3] == 1   # stockout at t+1
    assert np.isnan(out[U.TARGET].iloc[4])  # nothing observable after the last row


def test_target_is_nan_across_date_gaps():
    df = _toy([0] * 10).drop(index=[4])            # remove a calendar day
    out = add_target(df, horizon=3)
    assert out[U.TARGET].isna().sum() > 3


def test_missing_future_flag_gives_unknown_label_not_zero():
    """Regression: a blank stockout_flag used to be silently treated as 'no stockout'."""
    flags = [0.0] * 12
    flags[5] = np.nan
    out = add_target(_toy(flags), horizon=3)
    # days 2,3,4 have the missing day 5 inside their window -> unknown
    assert out[U.TARGET].iloc[2:5].isna().all()
    # ...but a stockout that IS observed in the window still makes the label 1
    flags[4] = 1.0
    out2 = add_target(_toy(flags), horizon=3)
    assert out2[U.TARGET].iloc[2] == 1


def test_rows_with_missing_inputs_are_excluded_not_filled(raw_df):
    df = raw_df.copy()
    df.loc[40, U.CLOSING_STOCK] = np.nan
    key = df.loc[40, [U.DATE, U.PRODUCT_ID, U.WAREHOUSE]]
    f = usable_rows(build_features(df))
    hit = (f[U.DATE] == key[U.DATE]) & (f[U.PRODUCT_ID] == key[U.PRODUCT_ID]) & (f[U.WAREHOUSE] == key[U.WAREHOUSE])
    assert not hit.any()


def test_target_respects_series_boundaries():
    a = _toy([0, 0, 0, 0, 0, 0]); b = _toy([1, 1, 1, 1, 1, 1]); b[U.PRODUCT_ID] = "P2"
    out = add_target(pd.concat([a, b]), horizon=2)
    assert (out[out[U.PRODUCT_ID] == "P1"][U.TARGET].dropna() == 0).all()


def test_target_prevalence_is_plausible(raw_df):
    out = labelled_rows(add_target(raw_df))
    assert 0.0 < out[U.TARGET].mean() < 0.6
    assert set(out[U.TARGET].unique()) <= {0.0, 1.0}


def test_target_horizon_is_configurable(raw_df):
    p3 = add_target(raw_df, horizon=3)[U.TARGET].mean()
    p14 = add_target(raw_df, horizon=14)[U.TARGET].mean()
    assert p14 >= p3                    # longer window can only catch more stockouts


# ---------------------------------------------------------------- features
def test_feature_columns_present_and_no_target_or_ids(raw_df):
    f = build_features(raw_df)
    assert set(FEATURE_COLUMNS) <= set(f.columns)
    banned = {U.TARGET, U.PRODUCT_ID, U.PRODUCT_NAME, U.DATE}
    assert banned.isdisjoint(FEATURE_COLUMNS)


def test_feature_values_known_case():
    n = 30
    df = pd.DataFrame({U.DATE: pd.date_range("2025-01-01", periods=n), U.PRODUCT_ID: "P", U.WAREHOUSE: "W",
                       U.CATEGORY: "C", U.UNITS_SOLD: 10, U.CLOSING_STOCK: 50, U.LEAD_TIME: 3,
                       U.REORDER_POINT: 25, U.OPEN_ORDER_QTY: 0, U.STOCKOUT_FLAG: 0})
    df.loc[20, U.STOCKOUT_FLAG] = 1
    f = build_features(df).iloc[-1]
    assert f["avg_daily_demand_28d"] == 10
    assert abs(f["days_of_cover"] - 5) < 1e-3
    assert abs(f["cover_minus_lead_time"] - 2) < 1e-3
    assert abs(f["stock_to_reorder_ratio"] - 2) < 1e-3
    assert f["order_pending"] == 0
    assert f["days_since_last_stockout"] == 9      # day 20 -> day 29
    assert f["stockouts_last_28d"] == 1


def test_features_do_not_use_the_future(raw_df):
    """Scramble everything AFTER a cutoff date; features on/before it must not change."""
    cutoff = raw_df[U.DATE].sort_values().iloc[len(raw_df) // 2]
    base = build_features(raw_df)

    tampered = raw_df.copy()
    future = tampered[U.DATE] > cutoff
    rng = np.random.default_rng(0)
    for col in (U.UNITS_SOLD, U.CLOSING_STOCK, U.OPEN_ORDER_QTY, U.STOCKOUT_FLAG):
        tampered.loc[future, col] = rng.integers(0, 2 if col == U.STOCKOUT_FLAG else 5000, future.sum())
    changed = build_features(tampered)

    past_a = base[base[U.DATE] <= cutoff].reset_index(drop=True)
    past_b = changed[changed[U.DATE] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(past_a[FEATURE_COLUMNS], past_b[FEATURE_COLUMNS])


def test_usable_rows_drop_short_history(raw_df):
    f = build_features(raw_df)
    u = usable_rows(f)
    assert len(u) < len(f)
    assert (u["history_days"] >= U.MIN_HISTORY_DAYS).all()
    assert u[FEATURE_COLUMNS].notna().all().all()


def test_missing_optional_columns_get_neutral_defaults(raw_df):
    slim = raw_df.drop(columns=[U.WAREHOUSE, "order_age_days"])
    slim = slim[slim[U.PRODUCT_ID] == "SKU-001"].drop_duplicates([U.DATE, U.PRODUCT_ID])
    f = build_features(slim)
    assert (f[U.WAREHOUSE] == "All").all()
