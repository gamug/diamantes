"""Model validation: cross-validation and generalization analysis (issue #26).

The training pipeline (issue #24) fits and tunes a regressor, then scores it once
on the held-out test set. That single number does not show whether the model
*generalizes*. This module adds the missing evidence:

* **K-fold cross-validation** of the tuned configuration on the training set
  (:func:`cross_validate_model`);
* a **train vs. cross-validation vs. test** comparison for every regression
  metric (:func:`build_comparison`);
* an **under/over-fitting diagnosis** with concrete improvement actions
  (:func:`diagnose_fit`);
* a **learning curve** (:func:`learning_curve_scores`) and summary plots as
  visual evidence.

:func:`validate_model` ties these together, writes a JSON report plus PNG plots
to ``data/08_reporting/`` and raises :class:`ModelValidationError` (or warns)
when cross-validated performance is too low or the train/CV gap is too wide.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.figure import Figure  # object-oriented API: no global backend, no pyplot state
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_validate, learning_curve

logger = logging.getLogger(__name__)

#: Directory for the validation report and plots (reporting layer).
REPORT_DIR: Path = Path(__file__).resolve().parents[1] / "data" / "08_reporting"

CV_FOLDS: int = 5
RANDOM_STATE: int = 42
N_JOBS: int = -1

#: Regression metrics -> the scikit-learn scoring key used for cross-validation.
#: All but ``r2`` are "``neg_``" scorers (higher = better), converted back to the
#: natural positive quantity by :func:`_as_positive`.
CV_SCORING: dict[str, str] = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "mape_pct": "neg_mean_absolute_percentage_error",
}

#: Cross-validated R^2 must reach this to be considered usable.
MIN_ACCEPTABLE_CV_R2: float = 0.80
#: ``train_r2 - cv_r2`` above this is *flagged* as overfitting in the diagnosis.
OVERFIT_R2_GAP: float = 0.05
#: ``train_r2 - cv_r2`` above this *fails* validation (hard stop).
FAIL_OVERFIT_R2_GAP: float = 0.10
#: CV R^2 below this with a small train/CV gap points to underfitting.
UNDERFIT_CV_R2: float = 0.85

#: An ``(X, y)`` pair -- one partition handed to the validator.
Partition = tuple[pd.DataFrame, pd.Series]


class ModelValidationError(RuntimeError):
    """Raised when cross-validated performance or the train/CV gap is unacceptable."""


def _as_positive(metric: str, raw_score: float) -> float:
    """Turn a raw cross-validation score into the natural positive quantity."""
    value = raw_score if metric == "r2" else -raw_score
    return value * 100.0 if metric == "mape_pct" else value


def score_split(
    model: BaseEstimator, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    """Score a fitted model on one partition (same metrics as cross-validation)."""
    predictions = model.predict(features)
    return {
        "r2": float(r2_score(target, predictions)),
        "mae": float(mean_absolute_error(target, predictions)),
        "rmse": float(root_mean_squared_error(target, predictions)),
        "mape_pct": float(mean_absolute_percentage_error(target, predictions) * 100.0),
    }


def cross_validate_model(
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = CV_FOLDS,
) -> dict[str, Any]:
    """K-fold cross-validate ``estimator`` on the training set.

    The estimator is cloned first, so a fitted, tuned pipeline can be passed in
    directly and its hyper-parameters (not the defaults) are what gets validated.

    Args:
        estimator: Any regressor / pipeline; cloned before fitting.
        x_train: Training predictors.
        y_train: Training target.
        folds: Number of shuffled :class:`~sklearn.model_selection.KFold` splits.

    Returns:
        ``{"folds": int, "metrics": {name: {"train_mean", "train_std",
        "val_mean", "val_std", "val_per_fold"}}}``.
    """
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    raw = cross_validate(
        clone(estimator),
        x_train,
        y_train,
        cv=splitter,
        scoring=CV_SCORING,
        return_train_score=True,
        n_jobs=N_JOBS,
    )
    metrics: dict[str, Any] = {}
    for name in CV_SCORING:
        train_scores = np.array([_as_positive(name, v) for v in raw[f"train_{name}"]])
        val_scores = np.array([_as_positive(name, v) for v in raw[f"test_{name}"]])
        metrics[name] = {
            "train_mean": float(train_scores.mean()),
            "train_std": float(train_scores.std()),
            "val_mean": float(val_scores.mean()),
            "val_std": float(val_scores.std()),
            "val_per_fold": [float(score) for score in val_scores],
        }
    return {"folds": folds, "metrics": metrics}


def build_comparison(
    cv_result: dict[str, Any],
    train_scores: dict[str, float],
    test_scores: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Line up train / cross-validation / test values for every metric."""
    return {
        name: {
            "train": train_scores[name],
            "cv_val": cv_result["metrics"][name]["val_mean"],
            "test": test_scores[name],
        }
        for name in CV_SCORING
    }


def diagnose_fit(train_r2: float, cv_r2: float, test_r2: float) -> dict[str, Any]:
    """Classify the fit as under-fitting, over-fitting or good, with actions.

    Args:
        train_r2: R^2 on the full training set.
        cv_r2: Mean cross-validated (validation-fold) R^2.
        test_r2: R^2 on the held-out test set.

    Returns:
        ``{"verdict", "reasons", "recommended_actions"}``.
    """
    train_cv_gap = train_r2 - cv_r2
    cv_test_gap = cv_r2 - test_r2
    reasons: list[str] = []
    actions: list[str] = []

    if cv_r2 < UNDERFIT_CV_R2 and train_cv_gap <= OVERFIT_R2_GAP:
        verdict = "underfitting"
        reasons.append(
            f"cross-validated R^2 {cv_r2:.3f} is below the {UNDERFIT_CV_R2:.2f} target while "
            f"the train/CV gap is only {train_cv_gap:.3f} (both scores are low)"
        )
        actions += [
            "add or engineer more informative features",
            "increase model capacity (more boosting iterations, larger max_depth)",
            "reduce regularization (lower l2_regularization)",
        ]
    elif train_cv_gap > OVERFIT_R2_GAP:
        verdict = "overfitting"
        reasons.append(
            f"training R^2 exceeds cross-validated R^2 by {train_cv_gap:.3f} "
            f"(> {OVERFIT_R2_GAP:.2f})"
        )
        actions += [
            "train on more data",
            "strengthen regularization (l2_regularization, min_samples_leaf, smaller max_depth/max_iter)",
            "enable early stopping",
            "drop weak or noisy features",
        ]
    else:
        verdict = "good_fit"
        reasons.append(
            f"training, CV and test R^2 agree within {max(abs(train_cv_gap), abs(cv_test_gap)):.3f}"
        )
        actions.append("model generalizes well; no change required")

    if abs(cv_test_gap) > OVERFIT_R2_GAP:
        reasons.append(
            f"CV R^2 and test R^2 differ by {abs(cv_test_gap):.3f} (> {OVERFIT_R2_GAP:.2f}); "
            "re-check the train/test split"
        )

    return {"verdict": verdict, "reasons": reasons, "recommended_actions": actions}


def learning_curve_scores(
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = CV_FOLDS,
) -> dict[str, list[float]]:
    """R^2 learning curve: train vs. validation score against training-set size."""
    sizes, train_scores, val_scores = learning_curve(
        clone(estimator),
        x_train,
        y_train,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE),
        scoring="r2",
        n_jobs=N_JOBS,
    )
    return {
        "train_sizes": [int(size) for size in sizes],
        "train_r2": [float(row.mean()) for row in train_scores],
        "val_r2": [float(row.mean()) for row in val_scores],
    }


def _plot_comparison(comparison: dict[str, dict[str, float]], path: Path) -> None:
    """Grouped bar chart: train / CV / test for each metric (one subplot each)."""
    names = list(comparison)
    fig = Figure(figsize=(3.2 * len(names), 3.4), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, len(names)))
    for axis, name in zip(axes, names, strict=True):
        values = comparison[name]
        axis.bar(
            ["train", "cv", "test"],
            [values["train"], values["cv_val"], values["test"]],
            color=["#4c72b0", "#dd8452", "#55a868"],
        )
        axis.set_title(name)
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle("Train vs. cross-validation vs. test")
    fig.savefig(path, dpi=120)


def _plot_learning_curve(curve: dict[str, list[float]], path: Path) -> None:
    """Plot the R^2 learning curve (train and validation vs. training size)."""
    fig = Figure(figsize=(6.0, 4.0), layout="constrained")
    axis = fig.subplots()
    axis.plot(curve["train_sizes"], curve["train_r2"], marker="o", label="train R^2")
    axis.plot(curve["train_sizes"], curve["val_r2"], marker="o", label="CV R^2")
    axis.set_xlabel("training samples")
    axis.set_ylabel("R^2")
    axis.set_title("Learning curve")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.savefig(path, dpi=120)


def _log_comparison(comparison: dict[str, dict[str, float]]) -> None:
    logger.info("%-10s %12s %12s %12s", "metric", "train", "cv", "test")
    for name, values in comparison.items():
        logger.info(
            "%-10s %12.4f %12.4f %12.4f", name, values["train"], values["cv_val"], values["test"]
        )


def validate_model(
    model: BaseEstimator,
    train: Partition,
    test: Partition,
    *,
    on_failure: Literal["raise", "warn"] = "raise",
    report_dir: Path | None = REPORT_DIR,
) -> dict[str, Any]:
    """Cross-validate ``model`` and analyse how well it generalizes.

    Args:
        model: The fitted (tuned) estimator from the training pipeline.
        train: Training partition as an ``(X, y)`` pair.
        test: Test partition as an ``(X, y)`` pair.
        on_failure: ``"raise"`` (default) raises :class:`ModelValidationError`
            when validation fails; ``"warn"`` logs and warns instead.
        report_dir: Directory for ``model_validation.json`` and the PNG plots
            (plus a learning-curve analysis). ``None`` skips all file output and
            the learning curve.

    Returns:
        The validation report: cross-validation summary, the
        train/CV/test comparison, generalization gaps, the fit diagnosis, the
        learning curve (when ``report_dir`` is set), the thresholds used and a
        ``passed`` flag.

    Raises:
        ModelValidationError: If validation fails and ``on_failure="raise"``.
    """
    x_train, y_train = train
    x_test, y_test = test

    cv_result = cross_validate_model(model, x_train, y_train)
    train_scores = score_split(model, x_train, y_train)
    test_scores = score_split(model, x_test, y_test)
    comparison = build_comparison(cv_result, train_scores, test_scores)
    _log_comparison(comparison)

    cv_r2 = comparison["r2"]["cv_val"]
    train_r2 = comparison["r2"]["train"]
    test_r2 = comparison["r2"]["test"]
    diagnosis = diagnose_fit(train_r2, cv_r2, test_r2)
    logger.info("Fit diagnosis: %s -- %s", diagnosis["verdict"], "; ".join(diagnosis["reasons"]))

    failures: list[str] = []
    if cv_r2 < MIN_ACCEPTABLE_CV_R2:
        failures.append(f"cross-validated R^2 {cv_r2:.3f} < required {MIN_ACCEPTABLE_CV_R2:.2f}")
    if train_r2 - cv_r2 > FAIL_OVERFIT_R2_GAP:
        failures.append(
            f"train/CV R^2 gap {train_r2 - cv_r2:.3f} > allowed {FAIL_OVERFIT_R2_GAP:.2f} "
            "(overfitting)"
        )

    report: dict[str, Any] = {
        "cv": cv_result,
        "holdout": {"train": train_scores, "test": test_scores},
        "comparison": comparison,
        "generalization_gap": {
            "train_minus_cv_r2": train_r2 - cv_r2,
            "cv_minus_test_r2": cv_r2 - test_r2,
        },
        "diagnosis": diagnosis,
        "learning_curve": None,
        "thresholds": {
            "min_acceptable_cv_r2": MIN_ACCEPTABLE_CV_R2,
            "overfit_r2_gap": OVERFIT_R2_GAP,
            "fail_overfit_r2_gap": FAIL_OVERFIT_R2_GAP,
            "underfit_cv_r2": UNDERFIT_CV_R2,
        },
        "passed": not failures,
        "failures": failures,
    }

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        report["learning_curve"] = learning_curve_scores(model, x_train, y_train)
        _plot_comparison(comparison, report_dir / "model_validation.png")
        _plot_learning_curve(
            report["learning_curve"], report_dir / "model_validation_learning_curve.png"
        )
        (report_dir / "model_validation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        logger.info("Wrote model validation report and plots to %s", report_dir)

    if failures:
        message = "model validation failed:\n- " + "\n- ".join(failures)
        if on_failure == "raise":
            raise ModelValidationError(message)
        logger.warning(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    return report
