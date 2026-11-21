"""Shared constants and small helpers used across SmartStock AI.

Keeping every "magic number" here means the prediction horizon, the random
seed and the column names are defined exactly once.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_DATA_PATH = DATA_DIR / "sample_inventory.csv"
MODEL_PATH = MODELS_DIR / "trained_model.joblib"

# ---------------------------------------------------------- global config
RANDOM_SEED = 42

# Prediction horizon: the label asks "will this product stock out at any time
# in the NEXT `HORIZON_DAYS` days (t+1 ... t+H)?". Configurable per call.
HORIZON_DAYS = 7

# Rolling windows (in days) used for demand features. Both look BACKWARDS only.
SHORT_WINDOW = 7
LONG_WINDOW = 28

# A row needs this many days of history before its rolling features are
# considered reliable enough for training or prediction.
MIN_HISTORY_DAYS = LONG_WINDOW

# ------------------------------------------------------- column contract
# Canonical column names of the RAW inventory table (one row per
# date x product x warehouse, recorded at END of day).
DATE = "date"
PRODUCT_ID = "product_id"
PRODUCT_NAME = "product_name"
CATEGORY = "category"
WAREHOUSE = "warehouse"
UNITS_SOLD = "units_sold"
CLOSING_STOCK = "closing_stock"
LEAD_TIME = "lead_time_days"
REORDER_POINT = "reorder_point"
OPEN_ORDER_QTY = "open_order_qty"
STOCKOUT_FLAG = "stockout_flag"

# Target column (created by target_generation.py; never a model input).
TARGET = "stockout_within_horizon"

# Identifies one time series (one product stored in one warehouse).
SERIES_KEYS = [PRODUCT_ID, WAREHOUSE]


def format_pct(value: float, digits: int = 1) -> str:
    """Format a 0-1 fraction as a percentage string, e.g. 0.123 -> '12.3%'."""
    return f"{value * 100:.{digits}f}%"
