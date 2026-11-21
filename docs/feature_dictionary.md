# Feature Dictionary

## Raw input columns (one row per date × product × warehouse, end of day)

| Column | Required | Type | Meaning |
|---|---|---|---|
| `date` | yes | date | day of the snapshot |
| `product_id` | yes | text | SKU code (identifier only — **not** a model feature) |
| `product_name` | no | text | display name |
| `category` | yes | text | product category |
| `warehouse` | no | text | distribution location (defaults to `"All"` if absent) |
| `units_sold` | yes | ≥ 0 | units sold that day (censored by available stock) |
| `closing_stock` | yes | ≥ 0 | units on hand at end of day |
| `lead_time_days` | yes | ≥ 0 | nominal replenishment lead time |
| `reorder_point` | yes | ≥ 0 | reorder threshold in units |
| `open_order_qty` | yes | ≥ 0 | units currently on order (0 if none) |
| `order_age_days` | no | ≥ 0 | days since the open order was placed (0 if absent) |
| `stockout_flag` | yes | 0/1 | 1 if demand exceeded stock that day |
| `replenishment_received` | no | ≥ 0 | units received that day (simulator bookkeeping; not a feature) |
| `stockout_within_horizon` | training only | 0/1/NaN | **target**; never a feature |

## Engineered model features

All use data from day t and earlier only.

| Feature | Formula | Why it matters |
|---|---|---|
| `current_stock` | `closing_stock` | units available now |
| `avg_daily_demand_7d` | mean `units_sold`, last 7 days | recent sales rate |
| `avg_daily_demand_28d` | mean `units_sold`, last 28 days | baseline sales rate |
| `demand_cv_28d` | std / mean, last 28 days | demand variability |
| `demand_trend_ratio` | 7d avg / 28d avg | recent demand change (>1 = accelerating) |
| `days_of_cover` | `current_stock / avg_daily_demand_28d` (cap 365) | days of inventory remaining |
| `lead_time_days` | nominal lead time | time needed to replenish |
| `cover_minus_lead_time` | `days_of_cover − lead_time_days` | negative = stock may run out before a new order could arrive |
| `reorder_point` | as supplied | planning threshold |
| `stock_to_reorder_ratio` | `current_stock / reorder_point` (cap 50) | position relative to threshold |
| `open_order_qty` | as supplied | stock already coming |
| `open_order_cover_days` | `open_order_qty / avg_daily_demand_28d` (cap 365) | how much demand the open order covers |
| `order_pending` | 1 if `open_order_qty > 0` | replenishment in progress |
| `order_age_days` | as supplied | how long the order has been in transit |
| `stockouts_last_28d` | count of stockout days, last 28 days | recent stockout history |
| `days_since_last_stockout` | days since last `stockout_flag = 1` (cap 60) | recency of problems |
| `day_of_week` | 0 = Monday … 6 = Sunday | weekly demand pattern |
| `month` | 1–12 | seasonal pattern |
| `category` | one-hot encoded in the pipeline | category behaviour |
| `warehouse` | one-hot encoded in the pipeline | location behaviour |

Rows with fewer than 28 days of history are excluded (rolling windows not yet reliable).

## Inventory turnover

Not included: with ≤ 18 months of daily data, a turnover ratio is almost a re-expression of `days_of_cover` and adds no independent information.

## Supplier features

No supplier identifier or historical supplier delay is available in the synthetic data, so none are used. A real deployment should add them (e.g. rolling average of actual vs. nominal lead time).
