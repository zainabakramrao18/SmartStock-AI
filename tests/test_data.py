"""Tests: dataset generation, validation, missing data, mapping, uploads."""
import io

import numpy as np
import pandas as pd
import pytest

from src import utilities as U
from src.data_generator import generate_inventory_data
from src.data_loader import load_table
from src.data_validation import (REQUIRED_COLUMNS, apply_mapping, suggest_mapping,
                                 validate_inventory_data)


# ------------------------------------------------------------ generation
def test_generator_is_reproducible():
    a = generate_inventory_data(n_days=30, n_products=2, seed=1)
    b = generate_inventory_data(n_days=30, n_products=2, seed=1)
    c = generate_inventory_data(n_days=30, n_products=2, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a[U.UNITS_SOLD].equals(c[U.UNITS_SOLD])


def test_generator_shape_and_physical_sanity(raw_df):
    assert len(raw_df) == 3 * 3 * 120
    assert set(REQUIRED_COLUMNS) <= set(raw_df.columns)
    assert (raw_df[U.CLOSING_STOCK] >= 0).all()
    assert (raw_df[U.UNITS_SOLD] >= 0).all()
    assert set(raw_df[U.STOCKOUT_FLAG].unique()) <= {0, 1}
    assert raw_df.duplicated([U.DATE, U.PRODUCT_ID, U.WAREHOUSE]).sum() == 0


def test_generator_has_no_target_column(raw_df):
    assert U.TARGET not in raw_df.columns


def test_stock_balance_holds(raw_df):
    """closing(t) = closing(t-1) + received(t) - sold(t) for every series."""
    for _, g in raw_df.sort_values(U.DATE).groupby(U.SERIES_KEYS):
        expected = g[U.CLOSING_STOCK].shift(1) + g["replenishment_received"] - g[U.UNITS_SOLD]
        assert (expected.iloc[1:] == g[U.CLOSING_STOCK].iloc[1:]).all()


# ------------------------------------------------------------ validation
def test_valid_data_passes(raw_df):
    rep = validate_inventory_data(raw_df)
    assert rep.ok, rep.summary()
    assert rep.n_series == 9


def test_missing_required_column_is_error(raw_df):
    rep = validate_inventory_data(raw_df.drop(columns=[U.LEAD_TIME]))
    assert not rep.ok
    assert U.LEAD_TIME in rep.missing_columns


def test_empty_table_is_error():
    assert not validate_inventory_data(pd.DataFrame()).ok


def test_missing_values_are_reported_not_filled(raw_df):
    df = raw_df.copy()
    df.loc[df.index[:3], U.CLOSING_STOCK] = np.nan
    rep = validate_inventory_data(df)
    assert rep.missing_values[U.CLOSING_STOCK] == 3
    assert rep.ok  # tiny share -> warning
    assert any("missing" in w for w in rep.warnings)
    assert df[U.CLOSING_STOCK].isna().sum() == 3  # input not modified


def test_many_missing_values_is_error(raw_df):
    df = raw_df.copy()
    df.loc[df.sample(frac=0.2, random_state=0).index, U.UNITS_SOLD] = np.nan
    assert not validate_inventory_data(df).ok


def test_non_numeric_and_bad_dates_detected(raw_df):
    df = raw_df.copy()
    df[U.CLOSING_STOCK] = df[U.CLOSING_STOCK].astype(object)
    df[U.DATE] = df[U.DATE].astype(object)  # an uploaded file would hold raw strings
    df.loc[df.index[0], U.CLOSING_STOCK] = "many"
    df.loc[df.index[1], U.DATE] = "not-a-date"
    rep = validate_inventory_data(df)
    assert U.CLOSING_STOCK in rep.dtype_issues
    assert U.DATE in rep.dtype_issues
    assert not rep.ok


def test_negative_values_and_bad_flags(raw_df):
    df = raw_df.copy()
    df.loc[df.index[0], U.CLOSING_STOCK] = -5
    df.loc[df.index[1], U.STOCKOUT_FLAG] = 2
    rep = validate_inventory_data(df)
    assert not rep.ok
    assert any("below 0" in e for e in rep.errors)
    assert any("0 or 1" in e for e in rep.errors)


def test_duplicates_detected(raw_df):
    df = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
    rep = validate_inventory_data(df)
    assert rep.duplicate_rows == 5 and not rep.ok


def test_gaps_in_dates_are_error(raw_df):
    df = raw_df.drop(raw_df.index[raw_df[U.DATE] == raw_df[U.DATE].iloc[50]])
    rep = validate_inventory_data(df)
    assert not rep.ok and any("missing calendar days" in e for e in rep.errors)


# --------------------------------------------------------- column mapping
def test_mapping_suggestion_and_application(raw_df):
    renamed = raw_df.rename(columns={U.CLOSING_STOCK: "Stock On Hand", U.PRODUCT_ID: "SKU",
                                     U.LEAD_TIME: "Lead-Time", U.UNITS_SOLD: "Daily Sales"})
    m = suggest_mapping(renamed.columns)
    assert m[U.CLOSING_STOCK] == "Stock On Hand"
    assert m[U.PRODUCT_ID] == "SKU"
    assert m[U.LEAD_TIME] == "Lead-Time"
    assert m[U.UNITS_SOLD] == "Daily Sales"
    fixed = apply_mapping(renamed, m)
    assert U.CLOSING_STOCK in fixed.columns
    assert validate_inventory_data(fixed).ok


def test_unmatched_columns_map_to_none_and_double_mapping_rejected(raw_df):
    m = suggest_mapping(["foo", "bar"])
    assert all(v is None for v in m.values())
    with pytest.raises(ValueError):
        apply_mapping(raw_df, {U.CLOSING_STOCK: U.UNITS_SOLD, U.UNITS_SOLD: U.UNITS_SOLD})


# ----------------------------------------------------------- file loading
def test_load_csv_and_excel_bytes(raw_df):
    small = raw_df.head(20)
    csv = small.to_csv(index=False).encode()
    pd.testing.assert_frame_equal(load_table(csv, "x.csv"), pd.read_csv(io.BytesIO(csv)))
    buf = io.BytesIO()
    small.to_excel(buf, index=False)
    assert len(load_table(buf.getvalue(), "x.xlsx")) == 20


def test_load_rejects_bad_files():
    with pytest.raises(ValueError, match="Unsupported"):
        load_table(b"abc", "notes.txt")
    with pytest.raises(ValueError, match="no rows"):
        load_table(b"a,b\n", "empty.csv")
