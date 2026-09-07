"""Unit tests for the :mod:`split_validation` train/test split checks (issue #25)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from diamond_features import KNOWN_CATEGORIES
from split_validation import (
    CATEGORICAL_FEATURES,
    TARGET,
    Partition,
    TrainTestSplitValidationError,
    build_split_validation_suite,
    failed_conditions,
    validate_train_test_split,
)


def _feature_matrix(n_rows: int, seed: int, *, clarities: list[str] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "carat": rng.uniform(0.3, 2.5, n_rows).astype("float32"),
            "depth": rng.uniform(58.0, 65.0, n_rows).astype("float32"),
            "table": rng.uniform(54.0, 60.0, n_rows).astype("float32"),
            "volume": rng.uniform(30.0, 400.0, n_rows).astype("float32"),
            "cut": pd.Categorical(rng.choice(KNOWN_CATEGORIES["cut"], n_rows)),
            "color": pd.Categorical(rng.choice(KNOWN_CATEGORIES["color"], n_rows)),
            "clarity": pd.Categorical(rng.choice(clarities or KNOWN_CATEGORIES["clarity"], n_rows)),
        }
    )


def _target(n_rows: int, seed: int) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).uniform(400.0, 15000.0, n_rows), name=TARGET)


def _clean_split(n_rows: int = 400, seed: int = 0) -> tuple[Partition, Partition]:
    """A same-distribution random split with no shared rows."""
    features = _feature_matrix(n_rows, seed)
    target = _target(n_rows, seed + 1)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=seed
    )
    return (x_train, y_train), (x_test, y_test)


@pytest.fixture
def clean_split() -> tuple[Partition, Partition]:
    return _clean_split()


def _with_leaked_rows(train: Partition, test: Partition, n_leaked: int = 60) -> Partition:
    x_train, y_train = train
    x_test, y_test = test
    x = pd.concat([x_train, x_test.iloc[:n_leaked]], ignore_index=True)
    y = pd.concat([y_train, y_test.iloc[:n_leaked]], ignore_index=True)
    return x, y


# --- suite construction ---------------------------------------------


def test_suite_bundles_the_expected_leakage_and_drift_checks() -> None:
    names = {type(check).__name__ for check in build_split_validation_suite().checks.values()}

    assert names == {
        "DatasetsSizeComparison",
        "TrainTestSamplesMix",
        "NewCategoryTrainTest",
        "StringMismatchComparison",
        "FeatureDrift",
        "LabelDrift",
        "FeatureLabelCorrelationChange",
    }


# --- happy path ----------------------------------------------------


def test_accepts_a_clean_split(clean_split: tuple[Partition, Partition]) -> None:
    train, test = clean_split

    result = validate_train_test_split(train, test, report_path=None)

    assert failed_conditions(result) == []


def test_writes_an_html_report(tmp_path: Path, clean_split: tuple[Partition, Partition]) -> None:
    train, test = clean_split
    report = tmp_path / "reports" / "split.html"

    validate_train_test_split(train, test, report_path=report)

    assert report.is_file()
    assert report.stat().st_size > 0
    assert "html" in report.read_text(errors="ignore").lower()


def test_report_is_skipped_when_path_is_none(
    tmp_path: Path, clean_split: tuple[Partition, Partition]
) -> None:
    train, test = clean_split

    validate_train_test_split(train, test, report_path=None)

    assert list(tmp_path.iterdir()) == []


# --- leakage / distribution failures --------------------------


def test_detects_row_leakage(clean_split: tuple[Partition, Partition]) -> None:
    train, test = clean_split
    leaked_train = _with_leaked_rows(train, test)

    with pytest.raises(TrainTestSplitValidationError, match="Train Test Samples Mix"):
        validate_train_test_split(leaked_train, test, report_path=None)


def test_detects_label_drift(clean_split: tuple[Partition, Partition]) -> None:
    (x_train, y_train), (x_test, y_test) = clean_split
    drifted_test = (x_test, y_test * 2.0 + 6000.0)

    with pytest.raises(TrainTestSplitValidationError, match="Label Drift"):
        validate_train_test_split((x_train, y_train), drifted_test, report_path=None)


def test_detects_feature_drift(clean_split: tuple[Partition, Partition]) -> None:
    (x_train, y_train), (x_test, y_test) = clean_split
    drifted_x_test = x_test.copy()
    drifted_x_test["carat"] = (drifted_x_test["carat"] + 3.0).astype("float32")

    with pytest.raises(TrainTestSplitValidationError, match="Feature Drift"):
        validate_train_test_split((x_train, y_train), (drifted_x_test, y_test), report_path=None)


def test_detects_category_only_present_in_test() -> None:
    train = (_feature_matrix(360, 3, clarities=["SI1", "SI2", "VS1", "VS2"]), _target(360, 5))
    test = (_feature_matrix(120, 4, clarities=["SI1", "VS2", "IF"]), _target(120, 6))

    with pytest.raises(TrainTestSplitValidationError, match="New Category Train Test"):
        validate_train_test_split(train, test, report_path=None)


# --- warn instead of raise -----------------------------------


def test_on_failure_warn_reports_without_raising(clean_split: tuple[Partition, Partition]) -> None:
    train, test = clean_split
    leaked_train = _with_leaked_rows(train, test)

    with pytest.warns(UserWarning, match="train/test split validation failed"):
        result = validate_train_test_split(leaked_train, test, on_failure="warn", report_path=None)

    assert any("Samples Mix" in failure for failure in failed_conditions(result))


# --- helper -------------------------------------------------


def test_failed_conditions_lists_every_non_passing_check(
    clean_split: tuple[Partition, Partition],
) -> None:
    (x_train, y_train), (x_test, y_test) = clean_split

    result = validate_train_test_split(
        (x_train, y_train),
        (x_test, y_test * 3.0 + 9000.0),
        on_failure="warn",
        report_path=None,
    )
    failures = failed_conditions(result)

    assert failures
    assert all(": " in failure for failure in failures)
    assert set(CATEGORICAL_FEATURES) == {"cut", "color", "clarity"}
