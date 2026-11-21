"""Tests: split, training, tuning, evaluation metrics, persistence."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier

from src import utilities as U
from src.evaluate_model import (baseline_metrics, calibration_table, compute_metrics, interpret_metrics,
                                threshold_table)
from src.feature_engineering import FEATURE_COLUMNS
from src.train_model import (TrainConfig, load_model, prepare_training_data, save_model, time_split,
                             train_model, tune_hyperparameters, write_training_report)


# ------------------------------------------------------------------ config
@pytest.mark.parametrize("bad", [dict(criterion="mse"), dict(max_depth=0), dict(min_samples_split=1),
                                 dict(min_samples_leaf=0), dict(class_weight="heavy"), dict(ccp_alpha=-1),
                                 dict(test_fraction=0.9), dict(horizon=0), dict(threshold=1.5)])
def test_config_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        TrainConfig(**bad).validate()


def test_config_defaults_valid_and_tree_params_match():
    cfg = TrainConfig().validate()
    p = cfg.tree_params()
    assert p["criterion"] == "gini" and p["random_state"] == U.RANDOM_SEED
    assert p["class_weight"] == "balanced" and p["ccp_alpha"] == 0.0


# ------------------------------------------------------------------ split
def test_time_split_is_chronological_with_purge_gap(train_raw):
    data = prepare_training_data(train_raw)
    train, test, info = time_split(data, 0.2, 7)
    assert train[U.DATE].max() < test[U.DATE].min()
    gap = (test[U.DATE].min() - train[U.DATE].max()).days
    assert gap >= 7 + 1                                   # no training label reaches the test period
    assert info["n_train"] == len(train) and info["n_test"] == len(test)
    assert info["n_train"] + info["n_test"] < len(data)   # the purge gap really drops rows
    assert 0.1 < info["n_test"] / len(data) < 0.3


def test_no_training_label_window_overlaps_test_period(train_raw):
    data = prepare_training_data(train_raw)
    train, test, _ = time_split(data, 0.2, 7)
    last_label_day = train[U.DATE].max() + pd.Timedelta(days=7)
    assert last_label_day < test[U.DATE].min()


def test_prepared_rows_are_labelled_and_have_history(train_raw):
    data = prepare_training_data(train_raw)
    assert data[U.TARGET].notna().all()
    assert (data["history_days"] >= U.MIN_HISTORY_DAYS).all()


def test_prepare_raises_when_data_too_short():
    from src.data_generator import generate_inventory_data
    with pytest.raises(ValueError, match="No usable labelled rows"):
        prepare_training_data(generate_inventory_data(n_days=20, n_products=1))


# --------------------------------------------------------------- training
def test_training_bundle_contents(bundle):
    assert set(bundle) >= {"pipeline", "config", "feature_columns", "split_info", "evaluation",
                           "interpretation", "probability_note", "sklearn_version"}
    assert bundle["feature_columns"] == FEATURE_COLUMNS
    tree = bundle["pipeline"].named_steps["tree"]
    assert isinstance(tree, DecisionTreeClassifier)
    assert tree.get_depth() <= 4
    assert tree.random_state == U.RANDOM_SEED


def test_training_is_deterministic(train_raw):
    cfg = TrainConfig(max_depth=4, min_samples_leaf=20)
    a, b = train_model(train_raw, cfg), train_model(train_raw, cfg)
    assert a["evaluation"]["metrics"] == b["evaluation"]["metrics"]


def test_model_learns_something_beyond_chance(bundle):
    m = bundle["evaluation"]["metrics"]
    assert m["roc_auc"] > 0.65
    assert m["recall"] > 0.0 and m["n_positive"] > 0


def test_pruning_and_entropy_options_change_the_tree(train_raw):
    base = train_model(train_raw, TrainConfig(max_depth=8, min_samples_leaf=5))
    pruned = train_model(train_raw, TrainConfig(max_depth=8, min_samples_leaf=5, ccp_alpha=0.002))
    ent = train_model(train_raw, TrainConfig(max_depth=4, criterion="entropy"))
    assert pruned["pipeline"].named_steps["tree"].get_n_leaves() < base["pipeline"].named_steps["tree"].get_n_leaves()
    assert ent["pipeline"].named_steps["tree"].criterion == "entropy"


def test_class_weight_raises_recall(train_raw):
    plain = train_model(train_raw, TrainConfig(class_weight=None))["evaluation"]["metrics"]["recall"]
    weighted = train_model(train_raw, TrainConfig(class_weight="balanced"))["evaluation"]["metrics"]["recall"]
    assert weighted >= plain


def test_tuning_selects_valid_config_without_touching_test(train_raw):
    data = prepare_training_data(train_raw)
    train, _, _ = time_split(data)
    grid = {"tree__max_depth": [2, 3], "tree__min_samples_leaf": [20, 50]}
    cfg, table = tune_hyperparameters(train, TrainConfig(), grid, n_splits=3)
    assert cfg.max_depth in (2, 3) and cfg.min_samples_leaf in (20, 50)
    assert len(table) == 4 and table["rank_test_score"].min() == 1
    cfg.validate()


def test_train_with_tune_flag(train_raw):
    b = train_model(train_raw, TrainConfig(), tune=True,
                    grid={"tree__max_depth": [3, 4], "tree__min_samples_leaf": [20]})
    assert b["tuned"] and b["tuning_results"] is not None and b["config"]["max_depth"] in (3, 4)


# ------------------------------------------------------------- evaluation
def test_metrics_match_sklearn_exactly():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    pred = (p >= 0.5).astype(int)
    m = compute_metrics(y, pred, p)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    assert (m["true_negatives"], m["false_positives"], m["false_negatives"], m["true_positives"]) == (tn, fp, fn, tp)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert m["precision"] == pytest.approx(precision_score(y, pred))
    assert m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert tp + tn + fp + fn == 500


def test_metrics_with_single_class_do_not_crash():
    m = compute_metrics([0, 0, 0], [0, 0, 0], [0.1, 0.2, 0.1])
    assert m["roc_auc"] is None and m["recall"] == 0.0 and m["accuracy"] == 1.0


def test_bundle_metrics_are_recomputable_from_the_model(bundle, train_raw):
    """Metrics in the bundle must equal a fresh computation on the same hold-out rows."""
    data = prepare_training_data(train_raw)
    _, test, _ = time_split(data, 0.2, 7)
    proba = bundle["pipeline"].predict_proba(test[FEATURE_COLUMNS])[:, 1]
    y = test[U.TARGET].astype(int)
    assert bundle["evaluation"]["metrics"]["roc_auc"] == pytest.approx(roc_auc_score(y, proba))
    assert bundle["evaluation"]["metrics"]["recall"] == pytest.approx(recall_score(y, proba >= 0.5))


def test_baselines_threshold_and_calibration_tables(bundle, train_raw):
    ev = bundle["evaluation"]
    assert set(ev["baselines"]) == {"all_low_risk", "all_high_risk", "cover_lt_lead_time"}
    assert ev["baselines"]["all_low_risk"]["recall"] == 0.0
    assert ev["baselines"]["all_high_risk"]["recall"] == 1.0
    tt = threshold_table([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.4])
    assert tt["recall"].is_monotonic_decreasing           # higher threshold never raises recall
    cal = calibration_table([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.4], n_bins=4)
    assert cal["n"].sum() == 4
    assert {"roc_curve", "pr_curve", "classification_report"} <= set(ev)


def test_interpretation_text_uses_real_numbers(bundle):
    m = bundle["evaluation"]["metrics"]
    text = interpret_metrics(m, bundle["evaluation"]["baselines"])
    assert f"{m['true_positives']:,}" in text and "synthetic" in text and "accuracy" in text.lower()


# ------------------------------------------------------------ persistence
def test_save_and_load_roundtrip_gives_identical_predictions(bundle, train_raw, tmp_path):
    path = save_model(bundle, tmp_path / "m.joblib")
    loaded = load_model(path)
    data = prepare_training_data(train_raw).tail(300)
    a = bundle["pipeline"].predict_proba(data[FEATURE_COLUMNS])
    b = loaded["pipeline"].predict_proba(data[FEATURE_COLUMNS])
    np.testing.assert_array_equal(a, b)
    assert loaded["config"] == bundle["config"]


def test_load_errors_are_readable(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_model"):
        load_model(tmp_path / "missing.joblib")
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="could not be loaded"):
        load_model(bad)
    import joblib
    joblib.dump({"hello": 1}, tmp_path / "wrong.joblib")
    with pytest.raises(ValueError, match="not a SmartStock"):
        load_model(tmp_path / "wrong.joblib")


def test_load_rejects_model_with_different_features(bundle, tmp_path):
    import joblib
    stale = dict(bundle, feature_columns=FEATURE_COLUMNS[:-1])
    joblib.dump(stale, tmp_path / "stale.joblib")
    with pytest.raises(ValueError, match="different feature set"):
        load_model(tmp_path / "stale.joblib")


def test_training_report_is_written(bundle, tmp_path):
    p = write_training_report(bundle, tmp_path / "r.json")
    import json
    rep = json.loads(p.read_text())
    assert rep["metrics"]["recall"] == bundle["evaluation"]["metrics"]["recall"]
    assert "Synthetic" in rep["note"]


# ---------------------------------------------------------------- leakage
def test_no_identifier_or_target_among_model_inputs(bundle):
    banned = {U.TARGET, U.PRODUCT_ID, U.PRODUCT_NAME, U.DATE, "replenishment_received", U.STOCKOUT_FLAG}
    assert banned.isdisjoint(bundle["feature_columns"])
    names = list(bundle["pipeline"].named_steps["prep"].get_feature_names_out())
    assert not any(n.startswith(("product_id", "product_name")) for n in names)
