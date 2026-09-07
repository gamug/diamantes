"""Unit tests for the :mod:`model_validation` cross-validation / fit analysis (issue #26)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from diamond_features import KNOWN_CATEGORIES
from model_validation import (
    CV_SCORING,
    ModelValidationError,
    build_comparison,
    cross_validate_model,
    diagnose_fit,
    learning_curve_scores,
    score_split,
    validate_model,
)
from training_pipeline import build_model_pipeline

Split = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]

_CV_FOLDS = 3  # small fold count keeps the unit tests fast
_LEARNABLE_R2 = 0.8  # the synthetic price signal is clearly learnable
_LEARNING_CURVE_POINTS = 5  # np.linspace(0.2, 1.0, 5) in learning_curve_scores


def _xy(n_rows: int = 320, seed: int = 0, *, signal: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    clarity = rng.choice(KNOWN_CATEGORIES["clarity"], n_rows)
    carat = rng.uniform(0.3, 2.5, n_rows)
    volume = carat * 60.0 + rng.normal(0.0, 3.0, n_rows)
    features = pd.DataFrame(
        {
            "carat": carat.astype("float32"),
            "depth": rng.uniform(58.0, 65.0, n_rows).astype("float32"),
            "table": rng.uniform(54.0, 60.0, n_rows).astype("float32"),
            "volume": volume.astype("float32"),
            "cut": pd.Categorical(rng.choice(KNOWN_CATEGORIES["cut"], n_rows)),
            "color": pd.Categorical(rng.choice(KNOWN_CATEGORIES["color"], n_rows)),
            "clarity": pd.Categorical(clarity),
        }
    )
    clarity_rank = np.array([KNOWN_CATEGORIES["clarity"].index(c) for c in clarity], dtype=float)
    if signal:
        target = (
            3000.0 * carat**1.8
            + 200.0 * clarity_rank
            + 10.0 * volume
            + rng.normal(0.0, 150.0, n_rows)
        )
    else:
        target = rng.uniform(400.0, 15000.0, n_rows)
    return features, pd.Series(target, name="price")


def _split(*, signal: bool = True, seed: int = 0) -> Split:
    features, target = _xy(seed=seed, signal=signal)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=seed
    )
    return x_train, x_test, y_train, y_test


@pytest.fixture
def fitted_split() -> tuple[BaseEstimator, Split]:
    x_train, x_test, y_train, y_test = _split()
    model = build_model_pipeline().fit(x_train, y_train)
    return model, (x_train, x_test, y_train, y_test)


# --- cross-validation --------------------------------------------


def test_cross_validate_model_reports_train_and_validation_scores(
    fitted_split: tuple[BaseEstimator, Split],
) -> None:
    model, (x_train, _, y_train, _) = fitted_split

    result = cross_validate_model(model, x_train, y_train, folds=_CV_FOLDS)

    assert result["folds"] == _CV_FOLDS
    assert set(result["metrics"]) == set(CV_SCORING)
    for stats in result["metrics"].values():
        assert {"train_mean", "train_std", "val_mean", "val_std", "val_per_fold"} <= set(stats)
        assert len(stats["val_per_fold"]) == _CV_FOLDS
    assert result["metrics"]["r2"]["val_mean"] > _LEARNABLE_R2


def test_score_split_matches_sklearn(fitted_split: tuple[BaseEstimator, Split]) -> None:
    model, (x_train, _, y_train, _) = fitted_split

    scores = score_split(model, x_train, y_train)

    assert scores["r2"] == pytest.approx(r2_score(y_train, model.predict(x_train)))
    assert set(scores) == set(CV_SCORING)


def test_build_comparison_lines_up_train_cv_test() -> None:
    cv_result = {"metrics": {name: {"val_mean": 0.5} for name in CV_SCORING}}
    train = dict.fromkeys(CV_SCORING, 0.9)
    test = dict.fromkeys(CV_SCORING, 0.4)

    comparison = build_comparison(cv_result, train, test)

    assert comparison["r2"] == {"train": 0.9, "cv_val": 0.5, "test": 0.4}
    assert set(comparison) == set(CV_SCORING)


# --- fit diagnosis ----------------------------------------------


def test_diagnose_fit_recognizes_a_good_fit() -> None:
    diagnosis = diagnose_fit(train_r2=0.97, cv_r2=0.96, test_r2=0.96)

    assert diagnosis["verdict"] == "good_fit"
    assert diagnosis["recommended_actions"]


def test_diagnose_fit_flags_overfitting_with_actions() -> None:
    diagnosis = diagnose_fit(train_r2=0.99, cv_r2=0.80, test_r2=0.79)

    assert diagnosis["verdict"] == "overfitting"
    assert any("regulari" in action for action in diagnosis["recommended_actions"])


def test_diagnose_fit_flags_underfitting() -> None:
    diagnosis = diagnose_fit(train_r2=0.70, cv_r2=0.68, test_r2=0.69)

    assert diagnosis["verdict"] == "underfitting"
    assert any("feature" in action for action in diagnosis["recommended_actions"])


def test_diagnose_fit_reports_the_gap_when_cv_beats_train() -> None:
    diagnosis = diagnose_fit(train_r2=0.90, cv_r2=0.95, test_r2=0.95)

    assert diagnosis["verdict"] == "good_fit"
    assert "within 0.050" in diagnosis["reasons"][0]  # abs(train - cv), not a signed 0.000


# --- learning curve -------------------------------------------


def test_learning_curve_scores_returns_increasing_sizes(
    fitted_split: tuple[BaseEstimator, Split],
) -> None:
    model, (x_train, _, y_train, _) = fitted_split

    curve = learning_curve_scores(model, x_train, y_train, folds=_CV_FOLDS)

    assert list(curve) == ["train_sizes", "train_r2", "val_r2"]
    assert (
        len(curve["train_sizes"])
        == len(curve["train_r2"])
        == len(curve["val_r2"])
        == _LEARNING_CURVE_POINTS
    )
    assert curve["train_sizes"] == sorted(curve["train_sizes"])


# --- validate_model orchestration ---------------------------


def test_validate_model_writes_json_and_plots(
    tmp_path: Path, fitted_split: tuple[BaseEstimator, Split]
) -> None:
    model, (x_train, x_test, y_train, y_test) = fitted_split

    report = validate_model(model, (x_train, y_train), (x_test, y_test), report_dir=tmp_path)

    assert (tmp_path / "model_validation.json").is_file()
    assert (tmp_path / "model_validation.png").is_file()
    assert (tmp_path / "model_validation_learning_curve.png").is_file()
    assert report["passed"] is True
    assert set(report) >= {"cv", "holdout", "comparison", "generalization_gap", "diagnosis"}
    assert report["learning_curve"] is not None


def test_validate_model_without_report_dir_skips_io(
    tmp_path: Path, fitted_split: tuple[BaseEstimator, Split]
) -> None:
    model, (x_train, x_test, y_train, y_test) = fitted_split

    report = validate_model(model, (x_train, y_train), (x_test, y_test), report_dir=None)

    assert list(tmp_path.iterdir()) == []
    assert report["learning_curve"] is None
    assert report["passed"] is True


def test_validate_model_raises_when_cross_validation_is_poor() -> None:
    x_train, x_test, y_train, y_test = _split(signal=False)
    model = build_model_pipeline().fit(x_train, y_train)

    with pytest.raises(ModelValidationError, match="cross-validated"):
        validate_model(model, (x_train, y_train), (x_test, y_test), report_dir=None)


def test_validate_model_warn_mode_returns_a_failed_report() -> None:
    x_train, x_test, y_train, y_test = _split(signal=False)
    model = build_model_pipeline().fit(x_train, y_train)

    with pytest.warns(UserWarning, match="model validation failed"):
        report = validate_model(
            model, (x_train, y_train), (x_test, y_test), on_failure="warn", report_dir=None
        )

    assert report["passed"] is False
    assert report["failures"]
