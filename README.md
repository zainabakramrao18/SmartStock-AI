# SmartStock AI
**Predictive Stockout Risk Intelligence for FMCG Distribution**

> ⚠️ **Prototype — Synthetic Data / Experimental ML Model.** Student project (Multan, Pakistan). Not affiliated with, endorsed by, or validated on data from PepsiCo or any company.

SmartStock AI uses a **Decision Tree classifier** to flag products likely to **stock out within the next 7 days**, so inventory managers can decide where to look first. It is decision support only — it does not place orders or guarantee that stockouts are prevented.

## Project status

| Phase | Scope | Status |
|---|---|---|
| 1 | Architecture, methodology, environment | ✅ done |
| 2 | Synthetic data, validation, target, features | ✅ done (28 tests passing) |
| 3 | Train / evaluate / tune / save model, explainability | ✅ done (79 tests passing) |
| 4 | Streamlit dashboard (8 pages) | ⏳ |
| 5 | Reports, full test suite | ⏳ |
| 6 | Deployment, demo script | ⏳ |

## Quick start (Windows)

1. Install **Python 3.11+** from python.org (tick *"Add Python to PATH"*) and **Git**.
2. Open **PowerShell** and run:

```powershell
git clone https://github.com/zainabakramrao18/SmartStock-AI.git
cd SmartStock-AI
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Create the sample dataset and the trained model. They are generated rather than stored in the repo, and the fixed random seed makes the data identical every time:

```powershell
python -m src.data_generator
python -m src.train_model --tune
```

4. Run the tests:

```powershell
python -m pytest tests -q
```

## Train the model (Phase 3)

```powershell
python -m src.train_model            # default parameters, about 4 seconds
python -m src.train_model --tune     # time-aware grid search first, about 35 seconds
python -m src.train_model --help     # every option (depth, criterion, pruning, horizon, ...)
```

This writes `models/trained_model.joblib` (the model) and `models/training_report.json` (readable metrics).
Train on your own file with `--data path\to\your.csv` (it must use the column names in `docs/feature_dictionary.md`).

## Use the model from Python

```python
import pandas as pd
from src.train_model import load_model
from src.prediction import predict_from_raw
from src.explainability import explain_row

model = load_model()
raw = pd.read_csv("data/sample_inventory.csv", parse_dates=["date"])
latest, info = predict_from_raw(model, raw, latest_only=True)     # one row per product-warehouse
top = latest.sort_values("risk_probability", ascending=False).head(1)
print(explain_row(model, top)["summary"])
```

macOS / Linux: use `python3 -m venv .venv && source .venv/bin/activate` instead of the activate line above.

## Documentation
* [Methodology](docs/methodology.md) — target, horizon, features, validation design
* [Feature dictionary](docs/feature_dictionary.md)
* [Architecture](docs/architecture.md)
* [Limitations](docs/limitations.md)
* [Dataset description](data/dataset_description.md)

## Evidence levels
| Claim | Status |
|---|---|
| Pipeline works on synthetic data | demonstrable |
| Model metrics on synthetic hold-out | measured: recall 86 %, precision 34 % (see `docs/methodology.md`) |
| Real-world performance | **not measured** |
| Business impact | **not measured** |
