"""Training, tuning, model selection and persistence.

Run from the project root:

    python -m src.train_model                 # train with default parameters
    python -m src.train_model --tune          # quick grid search (~1 minute)
    python -m src.train_model --tune --full   # larger grid search
    python -m src.train_model --help          # all options

Validation design
-----------------
1. Only rows whose label is known are used.
2. TIME-AWARE split: the earliest ~80 % of dates train, the latest ~20 % test.
3. PURGE GAP: a training row at date t has a label that looks at t+1..t+H.
   Training rows are therefore cut off H days before the first test date so no
   training label contains outcomes from the test period.
4. Hyper-parameter search uses TimeSeriesSplit INSIDE the training period only
   (with the same purge gap). The test set is used exactly once, at the end.
5. The model that is evaluated is the model that is saved (no silent refit).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src import utilities as U
from src.evaluate_model import evaluate, interpret_metrics
from src.feature_engineering import FEATURE_COLUMNS, build_features, usable_rows
from src.preprocessing import VALID_CRITERIA, build_pipeline
from src.target_generation import add_target

PROBABILITY_NOTE = (
    "Risk probability is the class share in the decision-tree leaf (with class weights if enabled). "
    "It is a RISK SCORE, not a calibrated probability; see the calibration table in the evaluation."
)

# Grids for the optional hyper-parameter search (keys are pipeline parameter names).
GRID_QUICK = {
    "tree__criterion": ["gini", "entropy"],
    "tree__max_depth": [3, 4, 5, 6, 8],
    "tree__min_samples_leaf": [25, 100],
    "tree__class_weight": [None, "balanced"],
    "tree__ccp_alpha": [0.0, 0.0005],
}
GRID_FULL = {
    "tree__criterion": ["gini", "entropy"],
    "tree__max_depth": [3, 4, 5, 6, 7, 8, 10, 12],
    "tree__min_samples_split": [20, 100],
    "tree__min_samples_leaf": [10, 25, 50, 100, 200],
    "tree__class_weight": [None, "balanced"],
    "tree__ccp_alpha": [0.0, 0.0002, 0.0005, 0.001],
}


@dataclass
class TrainConfig:
    """User-adjustable training settings (see the model-parameter section of the UI)."""

    criterion: str = "gini"
    max_depth: int | None = 5
    min_samples_split: int = 50
    min_samples_leaf: int = 25
    class_weight: str | None = "balanced"
    ccp_alpha: float = 0.0          # >0 enables cost-complexity pruning
    random_state: int = U.RANDOM_SEED
    horizon: int = U.HORIZON_DAYS
    test_fraction: float = 0.2
    threshold: float = 0.5          # score at/above which a row is called High Risk

    def validate(self) -> "TrainConfig":
        if self.criterion not in VALID_CRITERIA:
            raise ValueError(f"criterion must be one of {VALID_CRITERIA}")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be >= 1 (or None for unlimited)")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'")
        if self.ccp_alpha < 0:
            raise ValueError("ccp_alpha must be >= 0")
        if not 0.05 <= self.test_fraction <= 0.5:
            raise ValueError("test_fraction must be between 0.05 and 0.5")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1 day")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        return self

    def tree_params(self) -> dict:
        return dict(criterion=self.criterion, max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                    class_weight=self.class_weight, ccp_alpha=self.ccp_alpha,
                    random_state=self.random_state)


# ------------------------------------------------------------------ data prep
def prepare_training_data(df: pd.DataFrame, horizon: int = U.HORIZON_DAYS) -> pd.DataFrame:
    """Raw (or already-featured) table -> usable, labelled, date-sorted rows.

    Adds features if missing and ALWAYS rebuilds the target for the requested
    horizon (cheap, and guarantees the label matches the configured horizon),
    then drops rows with too little history, incomplete features or an unknown label.
    """
    out = df
    if "history_days" not in out.columns:
        out = build_features(out)
    out = add_target(out, horizon=horizon)
    out = usable_rows(out)
    out = out[out[U.TARGET].notna()].copy()
    if out.empty:
        raise ValueError("No usable labelled rows: need at least "
                         f"{U.MIN_HISTORY_DAYS + horizon + 1} days of data per product.")
    return out.sort_values([U.DATE] + U.SERIES_KEYS).reset_index(drop=True)


def time_split(df: pd.DataFrame, test_fraction: float = 0.2, horizon: int = U.HORIZON_DAYS):
    """Chronological train/test split with a purge gap of `horizon` days.

    Returns (train, test, info). Train rows end `horizon + 1` days before the first
    test date, so no training label overlaps the test period.
    """
    dates = np.sort(df[U.DATE].unique())
    if len(dates) < 10:
        raise ValueError("Too few distinct dates to split chronologically.")
    test_start = pd.Timestamp(dates[int(len(dates) * (1 - test_fraction))])
    train_end = test_start - pd.Timedelta(days=horizon + 1)
    train = df[df[U.DATE] <= train_end].copy()
    test = df[df[U.DATE] >= test_start].copy()
    if train.empty or test.empty:
        raise ValueError("Split produced an empty train or test set; use more data or a smaller test_fraction.")
    info = {
        "train_start": str(train[U.DATE].min().date()), "train_end": str(train[U.DATE].max().date()),
        "test_start": str(test[U.DATE].min().date()), "test_end": str(test[U.DATE].max().date()),
        "purge_gap_days": int(horizon), "n_train": int(len(train)), "n_test": int(len(test)),
        "train_positive_rate": float(train[U.TARGET].mean()), "test_positive_rate": float(test[U.TARGET].mean()),
    }
    return train, test, info


# ----------------------------------------------------------------- tuning
def tune_hyperparameters(train_df: pd.DataFrame, config: TrainConfig, grid: dict | None = None,
                         n_splits: int = 3, beta: float = 2.0) -> tuple[TrainConfig, pd.DataFrame]:
    """Grid search with time-ordered cross-validation inside the training period.

    Scored with F-beta (beta=2 weights recall twice as much as precision), matching
    the operational priority of not missing stockouts. Returns the best config and
    a table of all candidates.
    """
    train_df = train_df.sort_values(U.DATE).reset_index(drop=True)
    n_series = max(train_df.groupby(U.SERIES_KEYS).ngroups, 1)
    cv = TimeSeriesSplit(n_splits=n_splits, gap=config.horizon * n_series)  # gap counted in rows
    search = GridSearchCV(
        build_pipeline(config.tree_params()),
        param_grid=grid or GRID_QUICK,
        scoring=make_scorer(fbeta_score, beta=beta, zero_division=0),
        cv=cv, refit=False, n_jobs=1, error_score="raise",
    )
    search.fit(train_df[FEATURE_COLUMNS], train_df[U.TARGET].astype(int))
    res = pd.DataFrame(search.cv_results_)
    keep = ["rank_test_score", "mean_test_score", "std_test_score"] + [c for c in res if c.startswith("param_")]
    res = res[keep].sort_values("rank_test_score").reset_index(drop=True)

    best = search.best_params_
    tuned = TrainConfig(**{**asdict(config), **{k.replace("tree__", ""): v for k, v in best.items()}})
    return tuned.validate(), res


# ----------------------------------------------------------------- training
def train_model(df: pd.DataFrame, config: TrainConfig | None = None, tune: bool = False,
                grid: dict | None = None, data_source: str = "synthetic sample data") -> dict:
    """Run the whole pipeline and return a model bundle (a plain dict).

    The bundle contains the fitted pipeline plus everything needed to reproduce,
    display and audit it.
    """
    config = (config or TrainConfig()).validate()
    data = prepare_training_data(df, config.horizon)
    train, test, split_info = time_split(data, config.test_fraction, config.horizon)

    tuning = None
    if tune:
        config, tuning = tune_hyperparameters(train, config, grid)

    pipeline = build_pipeline(config.tree_params())
    pipeline.fit(train[FEATURE_COLUMNS], train[U.TARGET].astype(int))

    ev = evaluate(pipeline, test, FEATURE_COLUMNS, config.threshold)
    return {
        "pipeline": pipeline,
        "config": asdict(config),
        "feature_columns": list(FEATURE_COLUMNS),
        "split_info": split_info,
        "evaluation": ev,
        "interpretation": interpret_metrics(ev["metrics"], ev["baselines"]),
        "tuning_results": tuning,
        "tuned": bool(tune),
        "data_source": data_source,
        "probability_note": PROBABILITY_NOTE,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }


# -------------------------------------------------------------- persistence
def save_model(bundle: dict, path: Path | str = U.MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


_REQUIRED_KEYS = {"pipeline", "config", "feature_columns", "split_info", "evaluation"}


def load_model(path: Path | str = U.MODEL_PATH) -> dict:
    """Load a saved bundle, with readable errors for the common failure modes."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run:  python -m src.train_model")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ValueError(f"Model file {path.name} could not be loaded ({exc}). "
                         "Retrain with:  python -m src.train_model") from exc
    if not isinstance(bundle, dict) or not _REQUIRED_KEYS <= set(bundle):
        raise ValueError("This file is not a SmartStock AI model bundle. Retrain with:  python -m src.train_model")
    if bundle["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("The saved model was trained with a different feature set than this code version. "
                         "Retrain with:  python -m src.train_model")
    return bundle


def write_training_report(bundle: dict, path: Path | str) -> Path:
    """Human-readable JSON summary (metrics, parameters, split) stored beside the model."""
    ev = bundle["evaluation"]
    report = {
        "created_utc": bundle["created_utc"], "data_source": bundle["data_source"],
        "tuned": bundle["tuned"], "config": bundle["config"], "split": bundle["split_info"],
        "metrics": ev["metrics"], "baselines": ev["baselines"],
        "interpretation": bundle["interpretation"], "probability_note": bundle["probability_note"],
        "note": "Synthetic data. Not validated on any real distribution data.",
    }
    path = Path(path)
    path.write_text(json.dumps(report, indent=2))
    return path


# --------------------------------------------------------------------- CLI
def _cli() -> None:
    ap = argparse.ArgumentParser(description="Train the SmartStock AI decision tree.")
    ap.add_argument("--data", default=str(U.SAMPLE_DATA_PATH), help="CSV to train on (default: sample data)")
    ap.add_argument("--tune", action="store_true", help="run a time-aware grid search first")
    ap.add_argument("--full", action="store_true", help="with --tune: use the larger grid (slower)")
    ap.add_argument("--criterion", default="gini", choices=VALID_CRITERIA)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--min-samples-split", type=int, default=50)
    ap.add_argument("--min-samples-leaf", type=int, default=25)
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--ccp-alpha", type=float, default=0.0)
    ap.add_argument("--horizon", type=int, default=U.HORIZON_DAYS)
    ap.add_argument("--out", default=str(U.MODEL_PATH))
    a = ap.parse_args()

    cfg = TrainConfig(criterion=a.criterion, max_depth=a.max_depth, min_samples_split=a.min_samples_split,
                      min_samples_leaf=a.min_samples_leaf, class_weight=None if a.no_class_weight else "balanced",
                      ccp_alpha=a.ccp_alpha, horizon=a.horizon)
    df = pd.read_csv(a.data, parse_dates=[U.DATE])
    source = "synthetic sample data" if Path(a.data) == U.SAMPLE_DATA_PATH else f"user file {Path(a.data).name}"
    print(f"Training on {len(df):,} rows from {a.data} ...")
    bundle = train_model(df, cfg, tune=a.tune, grid=GRID_FULL if a.full else GRID_QUICK, data_source=source)

    out = save_model(bundle, a.out)
    rep = write_training_report(bundle, Path(a.out).with_name("training_report.json"))
    s, m = bundle["split_info"], bundle["evaluation"]["metrics"]
    print(f"\nTrain {s['train_start']} -> {s['train_end']} ({s['n_train']:,} rows, {s['train_positive_rate']:.1%} positive)")
    print(f"Test  {s['test_start']} -> {s['test_end']} ({s['n_test']:,} rows, {s['test_positive_rate']:.1%} positive)")
    print(f"Purge gap between them: {s['purge_gap_days']} days")
    print(f"Parameters: {bundle['config']}")
    print(f"\nHOLD-OUT RESULTS (stockout class)\n  recall {m['recall']:.3f} | precision {m['precision']:.3f} | "
          f"F1 {m['f1']:.3f} | accuracy {m['accuracy']:.3f} | ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f}")
    print(f"  TP {m['true_positives']}  FP {m['false_positives']}  FN {m['false_negatives']}  TN {m['true_negatives']}")
    print(f"\n{bundle['interpretation']}\n\nSaved model  -> {out}\nSaved report -> {rep}")


if __name__ == "__main__":
    _cli()
