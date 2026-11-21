"""Synthetic FMCG inventory generator.

IMPORTANT: this is SIMULATED data. It is not PepsiCo data and is not
validated against any real operation. Product names are generic on purpose.

How the simulation works (one loop iteration = one day, all series at once)
--------------------------------------------------------------------------
1. Any open replenishment order whose (actual) arrival day has come is
   received into stock.
2. True customer demand is drawn from a Gamma-Poisson (negative binomial)
   distribution whose mean = base demand x weekday effect x seasonal effect
   x slow trend x occasional promotion spike.
3. Sales = min(stock, demand). If demand > stock the day is a STOCKOUT
   (lost sales). Only sales are recorded - true demand is censored, exactly
   as it would be in a real sales system.
4. At end of day a simple (s, S) policy places an order when
   (stock + on-order) <= reorder_point and nothing is already on order.
5. The actual lead time = nominal lead time + a random supplier delay.

Why stockouts occur realistically (not by construction of a feature)
--------------------------------------------------------------------
* The planner's reorder point is based on a STALE demand estimate, so it is
  too low when demand rises (summer, promotions, trend).
* Supplier delays are random and the model cannot see them in advance.
These sources of noise mean the label is genuinely uncertain, so the model
cannot simply memorise it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import utilities as U

# (name, category, base units/day range, package seasonality amplitude)
_PRODUCTS = [
    ("Cola 500ml", "Carbonated Soft Drinks", (260, 520), 0.28),
    ("Cola 1.5L", "Carbonated Soft Drinks", (180, 380), 0.28),
    ("Cola 250ml Can", "Carbonated Soft Drinks", (90, 200), 0.25),
    ("Lemon-Lime Soda 500ml", "Carbonated Soft Drinks", (150, 320), 0.30),
    ("Lemon-Lime Soda 1.5L", "Carbonated Soft Drinks", (100, 240), 0.30),
    ("Orange Drink 250ml", "Juices & Nectars", (80, 190), 0.22),
    ("Mango Nectar 1L", "Juices & Nectars", (60, 160), 0.30),
    ("Mixed Fruit Juice 1L", "Juices & Nectars", (40, 120), 0.15),
    ("Apple Juice 200ml", "Juices & Nectars", (70, 170), 0.12),
    ("Mineral Water 500ml", "Bottled Water", (300, 600), 0.35),
    ("Mineral Water 1.5L", "Bottled Water", (220, 480), 0.35),
    ("Mineral Water 6L", "Bottled Water", (60, 150), 0.30),
    ("Energy Drink 250ml", "Energy Drinks", (40, 130), 0.10),
    ("Energy Drink 500ml", "Energy Drinks", (25, 90), 0.10),
    ("Iced Tea Lemon 345ml", "Ready-to-Drink Tea", (50, 140), 0.25),
    ("Iced Tea Peach 345ml", "Ready-to-Drink Tea", (35, 110), 0.25),
    ("Potato Chips Salted 50g", "Salty Snacks", (150, 360), 0.03),
    ("Masala Crisps 30g", "Salty Snacks", (200, 450), 0.03),
    ("Corn Puffs 40g", "Salty Snacks", (120, 300), 0.02),
    ("Tortilla Chips 80g", "Salty Snacks", (45, 130), 0.03),
]

_WAREHOUSES = [
    ("Multan DC", 1.00),
    ("Lahore DC", 1.45),
    ("Faisalabad DC", 0.80),
]

# Weekday multipliers, Monday..Sunday (Fri/Sat busier).
_DOW_MULT = np.array([0.92, 0.93, 0.95, 1.00, 1.12, 1.15, 0.93])


def generate_inventory_data(
    n_days: int = 540,
    start_date: str = "2025-01-01",
    seed: int = U.RANDOM_SEED,
    n_products: int | None = None,
) -> pd.DataFrame:
    """Generate the RAW daily inventory table (no target column).

    Parameters
    ----------
    n_days : number of consecutive daily records per series.
    start_date : first date, ISO string.
    seed : random seed - the same seed always gives the same data.
    n_products : use only the first N products (handy for fast tests).

    Returns
    -------
    DataFrame with columns: date, product_id, product_name, category,
    warehouse, units_sold, closing_stock, lead_time_days, reorder_point,
    open_order_qty, order_age_days, replenishment_received, stockout_flag.
    """
    rng = np.random.default_rng(seed)
    products = _PRODUCTS[:n_products] if n_products else _PRODUCTS

    # ---- build the list of series (product x warehouse) and their parameters
    meta = []
    for p_idx, (name, cat, (lo, hi), amp) in enumerate(products):
        for w_name, w_scale in _WAREHOUSES:
            base = rng.uniform(lo, hi) * w_scale
            nominal_lt = int(rng.integers(2, 9))
            plan_demand = base * rng.uniform(0.85, 1.10)  # stale planning estimate
            safety_days = rng.uniform(1.5, 4.5)
            rop = int(np.ceil(plan_demand * (nominal_lt + safety_days)))
            order_up_to = int(np.ceil(rop + plan_demand * rng.uniform(8, 16)))
            meta.append(
                dict(
                    product_id=f"SKU-{p_idx + 1:03d}",
                    product_name=name,
                    category=cat,
                    warehouse=w_name,
                    base=base,
                    amp=amp,
                    phase=rng.uniform(-20, 20),
                    trend=rng.normal(0.0002, 0.0003),
                    dispersion=rng.uniform(6.0, 20.0),  # higher = less volatile
                    nominal_lt=nominal_lt,
                    p_delay=rng.uniform(0.04, 0.22),  # supplier unreliability
                    rop=rop,
                    order_up_to=order_up_to,
                )
            )
    m = pd.DataFrame(meta)
    n = len(m)

    base = m["base"].to_numpy(float)
    amp = m["amp"].to_numpy(float)
    phase = m["phase"].to_numpy(float)
    trend = m["trend"].to_numpy(float)
    disp = m["dispersion"].to_numpy(float)
    nominal_lt = m["nominal_lt"].to_numpy(int)
    p_delay = m["p_delay"].to_numpy(float)
    rop = m["rop"].to_numpy(int)
    order_up_to = m["order_up_to"].to_numpy(int)

    # ---- state variables
    stock = order_up_to.astype(float) * rng.uniform(0.6, 1.0, n)
    stock = np.floor(stock)
    open_qty = np.zeros(n, dtype=int)
    arrival_day = np.zeros(n, dtype=int)
    placed_day = np.zeros(n, dtype=int)
    promo_left = np.zeros(n, dtype=int)
    promo_mult = np.ones(n)

    dates = pd.date_range(start_date, periods=n_days, freq="D")
    out = {k: np.zeros((n_days, n)) for k in
           ("sold", "close", "open", "age", "recv", "so")}

    for t in range(n_days):
        # 1. receive orders
        arrive = (open_qty > 0) & (arrival_day <= t)
        received = np.where(arrive, open_qty, 0)
        stock = stock + received
        open_qty = np.where(arrive, 0, open_qty)

        # 2. demand
        doy = dates[t].dayofyear
        season = 1.0 + amp * np.sin(2 * np.pi * (doy - 80 + phase) / 365.0)
        start_promo = (promo_left == 0) & (rng.random(n) < 0.008)
        promo_left = np.where(start_promo, 3, promo_left)
        promo_mult = np.where(start_promo, rng.uniform(1.5, 2.2, n), promo_mult)
        pm = np.where(promo_left > 0, promo_mult, 1.0)
        promo_left = np.maximum(promo_left - 1, 0)
        mu = base * _DOW_MULT[dates[t].dayofweek] * season * (1 + trend * t) * pm
        lam = rng.gamma(shape=disp, scale=mu / disp)
        demand = rng.poisson(lam)

        # 3. sales (censored by stock) and stockout flag
        sold = np.minimum(stock, demand)
        stockout = demand > stock
        stock = stock - sold

        # 4. reorder decision at end of day
        position = stock + open_qty
        place = (open_qty == 0) & (position <= rop)
        qty = np.ceil(order_up_to - position).astype(int)
        delayed = rng.random(n) < p_delay
        delay = np.where(delayed, rng.integers(1, 5, n), 0)
        actual_lt = nominal_lt + delay
        open_qty = np.where(place, qty, open_qty)
        arrival_day = np.where(place, t + actual_lt, arrival_day)
        placed_day = np.where(place, t, placed_day)

        out["sold"][t] = sold
        out["close"][t] = stock
        out["open"][t] = open_qty
        out["age"][t] = np.where(open_qty > 0, t - placed_day, 0)
        out["recv"][t] = received
        out["so"][t] = stockout

    # ---- assemble long-format table
    frames = []
    for j in range(n):
        frames.append(
            pd.DataFrame(
                {
                    U.DATE: dates,
                    U.PRODUCT_ID: m.at[j, "product_id"],
                    U.PRODUCT_NAME: m.at[j, "product_name"],
                    U.CATEGORY: m.at[j, "category"],
                    U.WAREHOUSE: m.at[j, "warehouse"],
                    U.UNITS_SOLD: out["sold"][:, j].astype(int),
                    U.CLOSING_STOCK: out["close"][:, j].astype(int),
                    U.LEAD_TIME: int(nominal_lt[j]),
                    U.REORDER_POINT: int(rop[j]),
                    U.OPEN_ORDER_QTY: out["open"][:, j].astype(int),
                    "order_age_days": out["age"][:, j].astype(int),
                    "replenishment_received": out["recv"][:, j].astype(int),
                    U.STOCKOUT_FLAG: out["so"][:, j].astype(int),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values([U.DATE, U.PRODUCT_ID, U.WAREHOUSE]).reset_index(drop=True)


def generate_sample_csv(path=U.SAMPLE_DATA_PATH, **kwargs) -> pd.DataFrame:
    """Generate data, attach the target label, and write the sample CSV."""
    from src.target_generation import add_target  # local import avoids a cycle

    df = add_target(generate_inventory_data(**kwargs))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    out = generate_sample_csv()
    print(f"Wrote {len(out):,} rows to {U.SAMPLE_DATA_PATH}")
    print(out[U.TARGET].value_counts(dropna=False))
