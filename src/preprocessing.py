"""scikit-learn preprocessing + model pipeline.

* Numeric features pass through unchanged: decision trees split on thresholds,
  so scaling is unnecessary (and would only make explanations harder to read).
* Categorical features (category, warehouse) are one-hot encoded.
  `handle_unknown="ignore"` means a category never seen in training becomes
  all-zeros instead of crashing a prediction run.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES

VALID_CRITERIA = ("gini", "entropy")


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,  # one-hot columns are named e.g. "category_Bottled Water"
    )


def build_pipeline(tree_params: dict) -> Pipeline:
    """Preprocessor followed by a DecisionTreeClassifier configured with `tree_params`."""
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("tree", DecisionTreeClassifier(**tree_params)),
        ]
    )


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Names of the columns the tree actually sees (numeric + one-hot)."""
    return list(pipeline.named_steps["prep"].get_feature_names_out())


def original_feature_of(transformed_name: str) -> str:
    """Map 'category_Bottled Water' back to 'category'; numeric names map to themselves."""
    for cat in CATEGORICAL_FEATURES:
        if transformed_name.startswith(cat + "_"):
            return cat
    return transformed_name
