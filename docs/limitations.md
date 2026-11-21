# Limitations

1. **Synthetic data only.** The simulator encodes our own assumptions about demand, lead times and planning. Results say how the method behaves on that simulation, not how it would perform on real distribution data.
2. **No affiliation or validation.** This is a student project. It is not affiliated with, endorsed by, or validated on data from PepsiCo or any company. Product names are generic.
3. **Decision support, not automation.** The tool ranks products for human investigation. It does not place orders, optimise replenishment or guarantee stockouts will be prevented.
4. **Business impact is unmeasured.** No claim is made about lost sales avoided or cost saved.
5. **Distribution shift over time.** In the simulation, planners' reorder points are fixed while demand grows and peaks seasonally, so stockouts become more frequent later in the data (about 2–4 % of rows in winter vs. 20 %+ in the final months). A model trained on earlier periods will meet a higher stockout rate at test time. Real systems also drift; retraining and monitoring are required.
6. **Censored demand.** `units_sold` is capped by stock, so demand during stockouts is underestimated — also true in real sales data.
7. **Time-aware validation has limits.** A single chronological hold-out reflects one period only; with seasonality, results can differ for other seasons. Rows of the same product are correlated, so effective sample size is smaller than the row count.
8. **Single decision tree.** Trees are easy to explain but unstable (small data changes can change the structure) and have limited accuracy compared with ensembles.
9. **Feature importance is not causation.** It describes what the fitted tree used to split, not what causes stockouts.
10. **Daily, gap-free data required.** Uploaded files with missing calendar days are rejected rather than silently repaired.
11. **No supplier history or promotions calendar** in the feature set; a real deployment should add them.
12. **Uploads are not persisted** on free hosting; files live only for the user's session.
