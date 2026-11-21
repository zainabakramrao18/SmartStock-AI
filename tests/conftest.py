"""Shared fixtures. Small datasets keep the whole suite fast (a few seconds)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generator import generate_inventory_data  # noqa: E402
from src.train_model import TrainConfig, train_model  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    # 3 products x 3 warehouses x 120 days
    return generate_inventory_data(n_days=120, n_products=3, seed=7)


@pytest.fixture(scope="session")
def train_raw():
    # 6 products x 3 warehouses x 330 days: enough positives and a seasonal ramp for a real split
    return generate_inventory_data(n_days=330, n_products=6, seed=11)


@pytest.fixture(scope="session")
def bundle(train_raw):
    return train_model(train_raw, TrainConfig(max_depth=4, min_samples_leaf=20))
