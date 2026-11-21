"""Tests: prediction output, probabilities, unseen categories, explanations."""
import numpy as np
import pandas as pd
import pytest

from src import utilities as U
from src.explainability import (decision_path, explain_row, feature_importance, path_text_for_rows,
                                render_tree, step_is_satisfied, tree_summary, tree_text)
from src.feature_engineering import FEATURE_COLUMNS
from src.prediction import (HIGH, LOW, latest_snapshot, predict, predict_from_raw, prepare_for_prediction,
                            to_prediction_table)


@pytest.fixture(scope="module")
def preds(bundle, train_raw):
    return predict_from_raw(bundle, train_raw)[0]


# ---------------------------------------------------------------- predict
def test_prediction_columns_and_ranges(preds):
    assert {"risk_probability", "risk_class", "risk_label", "risk_rank"} <= set(preds.columns)
    assert preds["risk_probability"].between(0, 1).all()
    assert set(preds["risk_class"].unique()) <= {0, 1}
    assert set(preds["risk_label"].unique()) <= {HIGH, LOW}
    assert preds["risk_rank"].min() == 1 and preds["risk_rank"].is_unique


def test_label_is_consistent_with_probability_and_threshold(bundle, preds):
    thr = bundle["config"]["threshold"]
    assert ((preds["risk_probability"] >= thr) == (preds["risk_class"] == 1)).all()
    strict = predict(bundle, preds[FEATURE_COLUMNS + [U.DATE]], threshold=0.95)
    assert strict["risk_class"].sum() <= preds["risk_class"].sum()


def test_probabilities_come_from_the_trained_model(bundle, preds):
    direct = bundle["pipeline"].predict_proba(preds[FEATURE_COLUMNS])[:, 1]
    np.testing.assert_allclose(preds["risk_probability"].to_numpy(), direct)


def test_early_rows_are_excluded_not_filled(bundle, train_raw):
    usable, info = prepare_for_prediction(train_raw)
    assert info["rows_excluded"] > 0 and info["rows_usable"] == len(usable)
    assert info["rows_in"] == info["rows_usable"] + info["rows_excluded"]


def test_latest_snapshot_has_one_row_per_series(bundle, train_raw):
    out, _ = predict_from_raw(bundle, train_raw, latest_only=True)
    assert len(out) == 18 and out[U.DATE].nunique() == 1
    assert out[U.DATE].iloc[0] == train_raw[U.DATE].max()
    assert len(latest_snapshot(out)) == 18


def test_unseen_category_does_not_crash(bundle, preds):
    row = preds.head(3).copy()
    row[U.CATEGORY] = "Brand New Category"
    row[U.WAREHOUSE] = "Unknown DC"
    out = predict(bundle, row)
    assert out["risk_probability"].between(0, 1).all()


def test_predict_rejects_missing_features_and_nans(bundle, preds):
    with pytest.raises(ValueError, match="feature columns missing"):
        predict(bundle, preds.drop(columns=["days_of_cover"]))
    bad = preds.head(2).copy()
    bad.loc[bad.index[0], "days_of_cover"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        predict(bundle, bad)
    with pytest.raises(ValueError, match="threshold"):
        predict(bundle, preds.head(2), threshold=1.2)


def test_empty_input_returns_empty_predictions(bundle, preds):
    out = predict(bundle, preds.head(0))
    assert len(out) == 0 and "risk_label" in out.columns


def test_prediction_table_layout(preds):
    t = to_prediction_table(preds.head(50))
    assert list(t.columns[:3]) == ["Product ID", "Product", "Category"]
    assert {"Stock", "Demand (units/day)", "Lead Time (days)", "Risk", "Risk Probability"} <= set(t.columns)
    assert t["Risk Probability"].is_monotonic_decreasing


# ------------------------------------------------------------ explanation
def _sample_rows(preds):
    hi = preds[preds["risk_class"] == 1].head(3)
    lo = preds[preds["risk_class"] == 0].head(3)
    assert len(hi) and len(lo), "fixture should contain both classes"
    return pd.concat([hi, lo])


def test_every_recorded_step_is_really_satisfied(bundle, preds):
    for i in range(len(_sample_rows(preds))):
        row = _sample_rows(preds).iloc[[i]]
        for step in decision_path(bundle, row)["steps"]:
            assert step_is_satisfied(step), step


def test_path_leaf_probability_equals_model_prediction(bundle, preds):
    for i in range(6):
        row = _sample_rows(preds).iloc[[i]]
        direct = bundle["pipeline"].predict_proba(row[FEATURE_COLUMNS])[0, 1]
        assert decision_path(bundle, row)["risk_probability"] == pytest.approx(direct)


def test_explanation_is_derived_from_real_values(bundle, preds):
    row = preds[preds["risk_class"] == 1].head(1)
    ex = explain_row(bundle, row)
    assert ex["predicted_label"] == HIGH
    assert ex["summary"].startswith("Predicted High Risk")
    assert ex["conditions"] and all(isinstance(c, str) for c in ex["conditions"])
    # each step quotes the product's actual value for that feature
    for step in ex["steps"]:
        if step["feature"] in FEATURE_COLUMNS and step["feature"] not in (U.CATEGORY, U.WAREHOUSE):
            assert float(row.iloc[0][step["feature"]]) == pytest.approx(step["value"], rel=1e-5, abs=1e-4)
    assert "not prove" in ex["limitations"]


def test_low_risk_row_is_labelled_low(bundle, preds):
    ex = explain_row(bundle, preds[preds["risk_class"] == 0].head(1))
    assert ex["predicted_label"] == LOW and ex["summary"].startswith("Predicted Low Risk")


def test_capped_value_is_not_shown_as_a_real_measurement():
    from src.explainability import _value_str
    assert _value_str("days_since_last_stockout", 60.0).startswith("60+")
    assert _value_str("days_since_last_stockout", 12.0) == "12.00"


def test_decision_path_requires_single_row(bundle, preds):
    with pytest.raises(ValueError):
        decision_path(bundle, preds.head(2))


def test_bulk_path_text_matches_row_count_and_is_nonempty(bundle, preds):
    txt = path_text_for_rows(bundle, preds.head(200))
    assert len(txt) == 200 and (txt.str.len() > 0).all()


def test_feature_importance_sums_to_one_and_is_grouped(bundle):
    imp = feature_importance(bundle)
    assert imp["importance"].sum() == pytest.approx(1.0)
    assert not imp["feature"].str.startswith("category_").any()
    assert imp["importance"].is_monotonic_decreasing
    raw = feature_importance(bundle, grouped=False)
    assert len(raw) >= len(imp)


def test_tree_rendering_and_text(bundle):
    fig = render_tree(bundle, max_depth=2)
    assert fig.axes
    txt = tree_text(bundle)
    assert "|---" in txt and "class:" in txt
    s = tree_summary(bundle)
    assert 1 <= s["depth"] <= 4 and s["n_leaves"] >= 2
