# Architecture

```
smartstock-ai/
├── app.py                     landing page (Phase 4)
├── pages/                     one Streamlit page per screen (Phase 4)
│   ├── 1_Overview.py   2_Predictions.py   3_High_Risk.py   4_Product_Explorer.py
│   └── 5_Model_Analytics.py   6_Explainability.py   7_Data_Management.py   8_Reports.py
├── src/                       all logic; pages contain only UI code
│   ├── utilities.py           constants: horizon, seed, windows, column names, paths      [done]
│   ├── data_generator.py      synthetic inventory simulator                               [done]
│   ├── data_loader.py         CSV / Excel reading                                         [done]
│   ├── data_validation.py     column spec, auto-mapping, validation report               [done]
│   ├── target_generation.py   future-window label                                         [done]
│   ├── feature_engineering.py backward-looking features                                   [done]
│   ├── preprocessing.py       sklearn ColumnTransformer (one-hot)                         [done]
│   ├── train_model.py         time-aware split, tuning, persistence                       [done]
│   ├── evaluate_model.py      metrics, curves                                             [done]
│   ├── prediction.py          load model, predict + probability                           [done]
│   ├── explainability.py      decision path → text, tree plot                             [done]
│   └── reporting.py           CSV / Excel / PDF exports                                   [Phase 5]
├── data/                      sample_inventory.csv, dataset_description.md
├── models/                    trained_model.joblib, training_report.json
├── tests/                     pytest suite
└── docs/                      methodology, feature dictionary, architecture, limitations
```

## Data flow

```
 upload / sample CSV
        │
 data_loader.load_table ──► suggest_mapping / apply_mapping ──► validate_inventory_data
        │                                                        (blocks prediction if errors)
        ▼
 feature_engineering.build_features ──► usable_rows
        │                                       │
        │ (training only)                       │ (prediction)
        ▼                                       ▼
 target_generation.add_target           prediction.predict  ──► explainability / reports / pages
        │
 train_model (time split + purge) ──► evaluate_model ──► models/trained_model.joblib
```

## Design decisions

* **Logic in `src/`, UI in `pages/`:** pages stay thin; everything is unit-testable without Streamlit.
* **One place for constants:** column names, horizon and seed live in `utilities.py`.
* **Validation gates prediction:** the UI will only enable "Run predictions" when `ValidationReport.ok` is true.
* **No database, no API keys:** plain files and pandas keep it lightweight on a low-spec Windows laptop.
* **Persistence caveat:** free hosts (e.g. Streamlit Community Cloud) have ephemeral disks. The trained model and sample data are committed to Git so they survive restarts; uploaded files are held in session memory only.
