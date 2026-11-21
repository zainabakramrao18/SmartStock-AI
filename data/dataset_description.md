# Dataset Description — `sample_inventory.csv`

**Synthetic data** produced by `src/data_generator.py` with `seed=42`. Regenerate with:

```
python -m src.data_generator
```

| Property | Value |
|---|---|
| Rows | 32,400 |
| Date range | 2025-01-01 → 2026-06-24 (540 consecutive days) |
| Products | 20 SKUs in 6 categories (generic names, no brands) |
| Warehouses | Multan DC, Lahore DC, Faisalabad DC |
| Series | 60 (20 products × 3 warehouses) |
| Labelled rows | 32,028 (last 7 days of each series have no observable future → blank target) |
| Positive rate | 2,523 / 32,028 ≈ 7.9 % of labelled rows have a stockout in the next 7 days |
| Daily stockout rate | ≈ 1.9 % of product-days |

## How stockouts arise
* Demand = base level × weekday effect × seasonal curve (stronger for water/soft drinks) × slow trend × occasional 3-day promotion spikes, drawn from a negative-binomial distribution.
* An (s, S) reorder policy runs per series with a **stale** demand estimate in its reorder point.
* Actual lead time = nominal lead time + random supplier delay (probability differs by series).
* Sales are capped at available stock, so recorded sales are censored, as in real systems.

## Columns
See `docs/feature_dictionary.md`. The `stockout_within_horizon` column is the **target**, created by `src/target_generation.py`; it is never used as a feature.

## Not real
All numbers are invented. Do not interpret any figure here as PepsiCo or market data.
