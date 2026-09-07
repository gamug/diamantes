"""Train/test split validation for the training pipeline (issue #25).

The training pipeline (issue #24) holds out a test set with
``training_pipeline.split_train_test``. Before a model is fitted we want
evidence that the split is sound:

* **no data leakage** -- the same stone must not appear in both partitions, and
  the feature/label relationship must not shift between them;
* **matching distributions** -- train and test must represent the same problem
  (no feature drift, label drift, unseen categories or degenerate sizes).

This module runs a curated `deepchecks` train/test suite to check exactly that.
It exposes one testable, pipeline-independent entry point,
:func:`validate_train_test_split`, which writes an HTML report of every check and
either raises :class:`TrainTestSplitValidationError` or emits a warning when a
condition fails.

``deepchecks`` 0.19.1 needs the shims in :mod:`_deepchecks_compat` to import on
this project's numpy / scikit-learn versions, so that module is imported first.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Literal

import pandas as pd

# ``deepchecks`` symbols come via the compat façade so its numpy / scikit-learn
# shims are always applied before ``deepchecks`` is imported (see the module).
from _deepchecks_compat import (
    CheckFailure,
    ConditionCategory,
    Dataset,
    DatasetsSizeComparison,
    FeatureDrift,
    FeatureLabelCorrelationChange,
    LabelDrift,
    NewCategoryTrainTest,
    StringMismatchComparison,
    Suite,
    SuiteResult,
    TrainTestSamplesMix,
)
from diamond_features import KNOWN_CATEGORIES, PRICE_COLUMN

logger = logging.getLogger(__name__)

#: Ordinal categorical predictors (kept in sync with the training pipeline).
CATEGORICAL_FEATURES: list[str] = list(KNOWN_CATEGORIES)
#: Regression target column.
TARGET: str = PRICE_COLUMN
#: Default location for the rendered validation report (reporting layer).
SPLIT_REPORT_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "08_reporting"
    / "train_test_split_validation.html"
)

# --- tolerances ------------------------------------------------------
#: A stone shared between train and test is leakage -- allow none.
MAX_SHARED_SAMPLES_RATIO: float = 0.0
#: Every category in the test set must have been seen in training.
MAX_NEW_CATEGORY_RATIO: float = 0.0
#: Per-feature drift ceilings (PSI for numeric, Cramer's V for categorical).
MAX_NUMERIC_DRIFT_SCORE: float = 0.2
MAX_CATEGORICAL_DRIFT_SCORE: float = 0.2
#: Target-distribution drift ceiling (Kolmogorov-Smirnov statistic).
MAX_LABEL_DRIFT_SCORE: float = 0.15
#: Allowed change in per-feature predictive power for the label between sets.
MAX_FEATURE_LABEL_PPS_DIFF: float = 0.2
#: Test set must be at least this fraction of the train set.
MIN_TEST_TRAIN_RATIO: float = 0.1


class TrainTestSplitValidationError(RuntimeError):
    """Raised when the train/test split fails a leakage or distribution check."""


def _to_dataset(features: pd.DataFrame, target: pd.Series, name: str) -> Dataset:
    """Wrap an ``(X, y)`` pair in a deepchecks regression :class:`Dataset`."""
    frame = features.copy()
    frame[TARGET] = target.to_numpy()
    categoricals = [column for column in CATEGORICAL_FEATURES if column in frame.columns]
    return Dataset(
        frame,
        label=TARGET,
        cat_features=categoricals,
        label_type="regression",
        dataset_name=name,
    )


def build_split_validation_suite() -> Suite:
    """Assemble the deepchecks suite used to vet a train/test split.

    Returns:
        A :class:`~deepchecks.tabular.Suite` of leakage and distribution checks,
        each carrying a pass/fail condition driven by the module tolerances.
    """
    return Suite(
        "Diamonds train/test split validation",
        DatasetsSizeComparison().add_condition_test_train_size_ratio_greater_than(
            MIN_TEST_TRAIN_RATIO
        ),
        TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(
            MAX_SHARED_SAMPLES_RATIO
        ),
        NewCategoryTrainTest().add_condition_new_category_ratio_less_or_equal(
            MAX_NEW_CATEGORY_RATIO
        ),
        StringMismatchComparison().add_condition_no_new_variants(),
        FeatureDrift().add_condition_drift_score_less_than(
            max_allowed_categorical_score=MAX_CATEGORICAL_DRIFT_SCORE,
            max_allowed_numeric_score=MAX_NUMERIC_DRIFT_SCORE,
        ),
        LabelDrift().add_condition_drift_score_less_than(MAX_LABEL_DRIFT_SCORE),
        FeatureLabelCorrelationChange().add_condition_feature_pps_difference_less_than(
            MAX_FEATURE_LABEL_PPS_DIFF
        ),
    )


def failed_conditions(result: SuiteResult) -> list[str]:
    """Return a message for every check that failed a condition or could not run."""
    failures: list[str] = []
    for check_result in result.results:
        if isinstance(check_result, CheckFailure):
            failures.append(
                f"{check_result.check.name()}: could not run -- {check_result.exception}"
            )
            continue
        for condition in check_result.conditions_results:
            if condition.category != ConditionCategory.PASS:
                failures.append(
                    f"{check_result.check.name()}: {condition.name} -- {condition.details}"
                )
    return failures


#: An ``(X, y)`` pair -- the shape of one partition handed to the validator.
Partition = tuple[pd.DataFrame, pd.Series]


def validate_train_test_split(
    train: Partition,
    test: Partition,
    *,
    on_failure: Literal["raise", "warn"] = "raise",
    report_path: Path | None = SPLIT_REPORT_PATH,
) -> SuiteResult:
    """Validate a train/test split for leakage and distribution mismatch.

    Args:
        train: The training partition as an ``(X, y)`` pair.
        test: The test partition as an ``(X, y)`` pair.
        on_failure: ``"raise"`` (default) raises
            :class:`TrainTestSplitValidationError` when any condition fails;
            ``"warn"`` logs a warning and issues a :class:`UserWarning` instead.
        report_path: Where to write the HTML report; ``None`` skips it.

    Returns:
        The full :class:`~deepchecks.core.SuiteResult`, so callers and notebooks
        can inspect every check regardless of pass/fail.

    Raises:
        TrainTestSplitValidationError: If a condition fails and
            ``on_failure="raise"``.
    """
    x_train, y_train = train
    x_test, y_test = test
    train_ds = _to_dataset(x_train, y_train, "train")
    test_ds = _to_dataset(x_test, y_test, "test")

    result = build_split_validation_suite().run(train_ds, test_ds)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        result.save_as_html(str(report_path), as_widget=False)
        logger.info("Wrote train/test split validation report to %s", report_path)

    failures = failed_conditions(result)
    if not failures:
        logger.info(
            "Train/test split passed all %d leakage/distribution checks", len(result.results)
        )
        return result

    message = "train/test split validation failed:\n- " + "\n- ".join(failures)
    if on_failure == "raise":
        raise TrainTestSplitValidationError(message)
    logger.warning(message)
    warnings.warn(message, UserWarning, stacklevel=2)
    return result
