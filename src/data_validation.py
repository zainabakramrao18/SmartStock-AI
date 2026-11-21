"""Validation and column mapping for inventory tables.

Principles
----------
* Never silently fabricate real-world values. Missing or invalid values are
  REPORTED; rows that cannot be used are excluded and counted, not filled.
* Every problem is described in plain language so a non-programmer can fix
  the file.
* Predictions are allowed only when `ValidationReport.ok` is True.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src import utilities as U

# ------------------------------------------------------------ column spec
# kind: "date" | "text" | "number" | "flag" (0/1)
# required: needed by the model pipeline.
COLUMN_SPEC: dict[str, dict] = {
    U.DATE: dict(kind="date", required=True,
                 desc="Date of the record (end-of-day snapshot)",
                 aliases=["date", "day", "record_date", "snapshot_date", "dt"]),
    U.PRODUCT_ID: dict(kind="text", required=True,
                       desc="Unique product / SKU code",
                       aliases=["product_id", "sku", "sku_id", "item_id", "item_code", "product_code"]),
    U.PRODUCT_NAME: dict(kind="text", required=False,
                         desc="Human-readable product name",
                         aliases=["product_name", "name", "item_name", "description", "sku_name"]),
    U.CATEGORY: dict(kind="text", required=True,
                     desc="Product category",
                     aliases=["category", "product_category", "segment", "item_category"]),
    U.WAREHOUSE: dict(kind="text", required=False,
                      desc="Warehouse / distribution location",
                      aliases=["warehouse", "location", "dc", "depot", "store", "site", "warehouse_id"]),
    U.UNITS_SOLD: dict(kind="number", required=True, min=0,
                       desc="Units sold during the day",
                       aliases=["units_sold", "sales", "qty_sold", "quantity_sold", "daily_sales", "sold"]),
    U.CLOSING_STOCK: dict(kind="number", required=True, min=0,
                          desc="Units in stock at END of day",
                          aliases=["closing_stock", "stock", "inventory", "on_hand", "stock_on_hand",
                                   "inventory_level", "current_stock"]),
    U.LEAD_TIME: dict(kind="number", required=True, min=0,
                      desc="Nominal replenishment lead time (days)",
                      aliases=["lead_time_days", "lead_time", "leadtime", "supplier_lead_time"]),
    U.REORDER_POINT: dict(kind="number", required=True, min=0,
                          desc="Reorder threshold (units)",
                          aliases=["reorder_point", "rop", "reorder_level", "reorder_threshold"]),
    U.OPEN_ORDER_QTY: dict(kind="number", required=True, min=0,
                           desc="Units currently on order (0 if none)",
                           aliases=["open_order_qty", "on_order", "open_orders", "pending_order_qty",
                                    "qty_on_order", "incoming_stock"]),
    U.STOCKOUT_FLAG: dict(kind="flag", required=True,
                          desc="1 if demand could not be fully met that day, else 0",
                          aliases=["stockout_flag", "stockout", "stock_out", "out_of_stock", "oos", "oos_flag"]),
}

REQUIRED_COLUMNS = [c for c, s in COLUMN_SPEC.items() if s["required"]]
OPTIONAL_COLUMNS = [c for c, s in COLUMN_SPEC.items() if not s["required"]]


@dataclass
class ValidationReport:
    """Outcome of validating a table. `ok` is False if any error exists."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)
    dtype_issues: dict[str, str] = field(default_factory=dict)
    duplicate_rows: int = 0
    n_rows: int = 0
    n_series: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        head = "PASSED" if self.ok else "FAILED"
        lines = [f"Validation {head}: {self.n_rows:,} rows, {self.n_series} product-location series."]
        lines += [f"ERROR: {e}" for e in self.errors]
        lines += [f"Warning: {w}" for w in self.warnings]
        return "\n".join(lines)


# ----------------------------------------------------------------- mapping
def suggest_mapping(columns) -> dict[str, str | None]:
    """Guess which uploaded column matches each canonical column.

    Matching is case/space/underscore-insensitive against known aliases.
    Returns {canonical_name: uploaded_column_or_None}. The user can always
    override the suggestion in the UI.
    """
    norm = {str(c).strip().lower().replace(" ", "_").replace("-", "_"): c for c in columns}
    used: set = set()
    mapping: dict[str, str | None] = {}
    for canon, spec in COLUMN_SPEC.items():
        found = None
        for alias in spec["aliases"]:
            if alias in norm and norm[alias] not in used:
                found = norm[alias]
                break
        if found is not None:
            used.add(found)
        mapping[canon] = found
    return mapping


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Rename/select columns to canonical names. Unmapped columns are omitted."""
    picked = {canon: up for canon, up in mapping.items() if up and up in df.columns}
    ups = list(picked.values())
    dupes = sorted({u for u in ups if ups.count(u) > 1})
    if dupes:
        raise ValueError(f"Uploaded column(s) {dupes} were mapped to more than one required field. "
                         "Each uploaded column can be used only once.")
    return df[ups].rename(columns={up: canon for canon, up in picked.items()}).copy()


# -------------------------------------------------------------- validation
def validate_inventory_data(df: pd.DataFrame) -> ValidationReport:
    """Validate a table that already uses canonical column names."""
    rep = ValidationReport(n_rows=len(df))

    if len(df) == 0:
        rep.errors.append("The table is empty.")
        return rep

    # 1. required columns present?
    rep.missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    for c in rep.missing_columns:
        rep.errors.append(f"Required column '{c}' is missing ({COLUMN_SPEC[c]['desc']}). "
                          "Map it to one of your columns or add it to the file.")
    if rep.missing_columns:
        return rep  # nothing else can be checked reliably

    work = df.copy()

    # 2. missing values (reported per column, never filled)
    for c in work.columns:
        if c in COLUMN_SPEC:
            n_missing = int(work[c].isna().sum())
            if n_missing:
                rep.missing_values[c] = n_missing
    for c, n in rep.missing_values.items():
        if COLUMN_SPEC[c]["required"]:
            pct = n / len(work)
            msg = f"Column '{c}' has {n:,} missing values ({pct:.1%}). Those rows will be excluded, not filled in."
            (rep.errors if pct > 0.05 else rep.warnings).append(
                msg + (" More than 5% missing - please fix the source data." if pct > 0.05 else ""))
        else:
            rep.warnings.append(f"Optional column '{c}' has {n:,} missing values.")

    # 3. data types
    parsed_date = pd.to_datetime(work[U.DATE], errors="coerce")
    bad_dates = int(parsed_date.isna().sum() - work[U.DATE].isna().sum())
    if bad_dates:
        rep.dtype_issues[U.DATE] = f"{bad_dates} values are not valid dates"
        rep.errors.append(f"Column '{U.DATE}' has {bad_dates:,} values that are not valid dates.")
    for c, spec in COLUMN_SPEC.items():
        if c not in work.columns or spec["kind"] not in ("number", "flag"):
            continue
        num = pd.to_numeric(work[c], errors="coerce")
        bad = int(num.isna().sum() - work[c].isna().sum())
        if bad:
            rep.dtype_issues[c] = f"{bad} non-numeric values"
            rep.errors.append(f"Column '{c}' must be numeric but has {bad:,} non-numeric values.")
            continue
        if "min" in spec and (num < spec["min"]).any():
            n_neg = int((num < spec["min"]).sum())
            rep.errors.append(f"Column '{c}' has {n_neg:,} values below {spec['min']} (not physically possible).")
        if spec["kind"] == "flag" and not num.dropna().isin([0, 1]).all():
            rep.errors.append(f"Column '{c}' must contain only 0 or 1.")
        work[c] = num
    work[U.DATE] = parsed_date

    # 4. duplicates on the natural key
    keys = [U.DATE, U.PRODUCT_ID] + ([U.WAREHOUSE] if U.WAREHOUSE in work.columns else [])
    rep.duplicate_rows = int(work.duplicated(subset=keys).sum())
    if rep.duplicate_rows:
        rep.errors.append(f"{rep.duplicate_rows:,} duplicate rows share the same date/product/location. "
                          "Remove duplicates before predicting.")

    if rep.errors:
        return rep

    # 5. history length and date continuity per series
    series_keys = [U.PRODUCT_ID] + ([U.WAREHOUSE] if U.WAREHOUSE in work.columns else [])
    g = work.groupby(series_keys)
    rep.n_series = g.ngroups
    span = g[U.DATE].agg(lambda s: (s.max() - s.min()).days + 1)
    count = g[U.DATE].count()
    short = int((count < U.MIN_HISTORY_DAYS).sum())
    gappy = int((span != count).sum())
    if short:
        rep.warnings.append(f"{short} series have fewer than {U.MIN_HISTORY_DAYS} days of history; "
                            "their rolling features are unreliable and they will be excluded from predictions.")
    if gappy:
        rep.errors.append(f"{gappy} series have missing calendar days. Daily records without gaps are required "
                          "because demand features use rolling windows. Fill the gaps from your source system.")

    # 6. logical consistency (warnings only)
    if (work[U.REORDER_POINT] == 0).all():
        rep.warnings.append("reorder_point is 0 everywhere - stock-to-reorder features will be uninformative.")
    return rep
