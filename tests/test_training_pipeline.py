"""Unit tests for the :mod:`training_pipeline` script (issue #24)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from joblib import load
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import NotFittedError

import training_pipeline as tp
from diamond_features import (
    FEATURE_TABLE_COLUMNS,
    KNOWN_CATEGORIES,
    cast_feature_dtypes,
)
from feature_pipeline import FEATURE_PARQUET_PATH
from training_pipeline import (
    MODEL_FEATURES,
    TARGET,
    build_model_pipeline,
    evaluate_model,
    load_feature_table,
    main,
    run_training_pipeline,
    save_metrics,
    save_model,
    split_features_target,
    split_train_test,
    train_model,
)

_NO_GRID: dict[str, list[object]] = {}  # skip tuning: fit the base pipeline once
_MIN_LEARNABLE_R2 = 0.5  # the synthetic price signal is clearly learnable
_SMOKE_MAX_MAPE_PCT = 25.0  # boosted trees comfortably beat this on real diamonds
_SMOKE_MIN_R2 = 0.9


def _make_feature_frame(n_rows: int = 240, seed: int = 0) -> pd.DataFrame:
    """A synthetic feature table shaped exactly like the feature-pipeline output.

    Price is a smooth function of ``carat``, ``volume`` and the ordinal grades
    plus noise, so a boosted-tree regressor can actually learn something.
    """
    rng = np.random.default_rng(seed)
    cut = rng.choice(KNOWN_CATEGORIES["cut"], n_rows)
    color = rng.choice(KNOWN_CATEGORIES["color"], n_rows)
    clarity = rng.choice(KNOWN_CATEGORIES["clarity"], n_rows)
    carat = rng.uniform(0.2, 3.0, n_rows)
    depth = rng.uniform(55.0, 67.0, n_rows)
    table = rng.uniform(52.0, 62.0, n_rows)
    x = np.cbrt(carat) * 4.1 + rng.normal(0.0, 0.05, n_rows)
    y = x + rng.normal(0.0, 0.02, n_rows)
    z = x * 0.62 + rng.normal(0.0, 0.02, n_rows)
    volume = x * y * z

    def _ranks(values: np.ndarray, order: list[str]) -> np.ndarray:
        lookup = {grade: rank for rank, grade in enumerate(order)}
        return np.array([lookup[value] for value in values], dtype=float)

    price = (
        3500.0 * carat**1.9
        + 180.0 * _ranks(cut, KNOWN_CATEGORIES["cut"])
        + 220.0 * _ranks(color, KNOWN_CATEGORIES["color"])
        + 260.0 * _ranks(clarity, KNOWN_CATEGORIES["clarity"])
        + 12.0 * volume
        + rng.normal(0.0, 150.0, n_rows)
    )
    frame = pd.DataFrame(
        {
            "carat": carat,
            "cut": cut,
            "color": color,
            "clarity": clarity,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z,
            "volume": volume,
            "price": price.clip(326, 18823).round().astype("int64"),
        }
    )[FEATURE_TABLE_COLUMNS]
    return cast_feature_dtypes(frame)


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    return _make_feature_frame()


@pytest.fixture
def feature_parquet(tmp_path: Path, feature_frame: pd.DataFrame) -> Path:
    path = tmp_path / "04_feature" / "diamantes_features.parquet"
    path.parent.mkdir(parents=True)
    feature_frame.to_parquet(path, index=False)
    return path


@pytest.fixture
def fitted_model(feature_frame: pd.DataFrame) -> TransformedTargetRegressor:
    features, target = split_features_target(feature_frame)
    return train_model(features, target, param_grid=_NO_GRID)


# --- data reading -------------------------------------------------------


def test_load_feature_table_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_feature_table(tmp_path / "nope.parquet")


def test_load_feature_table_rejects_missing_column(
    tmp_path: Path, feature_frame: pd.DataFrame
) -> None:
    path = tmp_path / "incomplete.parquet"
    feature_frame.drop(columns=["volume"]).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_feature_table(path)


def test_load_feature_table_returns_only_model_columns(feature_parquet: Path) -> None:
    loaded = load_feature_table(feature_parquet)

    assert list(loaded.columns) == [*MODEL_FEATURES, TARGET]
    assert not loaded.empty


def test_split_features_target_separates_predictors_and_label(feature_frame: pd.DataFrame) -> None:
    features, target = split_features_target(feature_frame)

    assert list(features.columns) == MODEL_FEATURES
    assert target.name == TARGET
    assert len(features) == len(target) == len(feature_frame)


# --- train / test split ---------------------------------------------


def test_split_train_test_is_deterministic_and_disjoint(feature_frame: pd.DataFrame) -> None:
    features, target = split_features_target(feature_frame)

    x_train, x_test, y_train, y_test = split_train_test(features, target, stratify=False)
    again = split_train_test(features, target, stratify=False)

    assert len(x_test) == pytest.approx(len(feature_frame) * tp.TEST_SIZE, abs=1)
    assert len(x_train) + len(x_test) == len(feature_frame)
    assert set(x_train.index).isdisjoint(x_test.index)
    assert list(y_train.index) == list(x_train.index)
    pd.testing.assert_frame_equal(x_train, again[0])
    pd.testing.assert_series_equal(y_test, again[3])


def test_split_train_test_stratified_runs_on_balanced_frame() -> None:
    frame = _make_feature_frame(n_rows=200)
    frame["cut"] = pd.Series(["Ideal", "Premium"] * 100, dtype="category")
    frame["clarity"] = pd.Series(["SI1", "VS2"] * 100, dtype="category")
    features, target = split_features_target(frame)

    x_train, x_test, _, _ = split_train_test(features, target, stratify=True)

    assert len(x_train) + len(x_test) == len(frame)


# --- model construction & training --------------------------------


def test_build_model_pipeline_returns_unfitted_estimator(feature_frame: pd.DataFrame) -> None:
    model = build_model_pipeline()
    features, _ = split_features_target(feature_frame)

    assert isinstance(model, TransformedTargetRegressor)
    with pytest.raises(NotFittedError):
        model.predict(features)


def test_train_model_without_grid_fits_base_pipeline(feature_frame: pd.DataFrame) -> None:
    features, target = split_features_target(feature_frame)

    model = train_model(features, target, param_grid=_NO_GRID)
    predictions = model.predict(features)

    assert len(predictions) == len(features)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()  # log-target transform keeps prices positive


def test_train_model_with_grid_runs_search_and_fits(feature_frame: pd.DataFrame) -> None:
    features, target = split_features_target(feature_frame)

    model = train_model(features, target, param_grid={"regressor__model__max_iter": [40, 80]})

    assert isinstance(model, TransformedTargetRegressor)
    assert np.isfinite(model.predict(features)).all()


def test_train_model_defaults_to_the_module_grid(
    feature_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_grid_search(estimator: object, param_grid: object, **kwargs: object) -> object:
        captured["grid"] = param_grid

        class _Stub:
            best_params_: ClassVar[dict[str, object]] = {}
            best_estimator_ = build_model_pipeline()

            def fit(self, *_: object, **__: object) -> None:
                self.best_estimator_.fit(feature_frame[MODEL_FEATURES], feature_frame[TARGET])

        return _Stub()

    monkeypatch.setattr(tp, "GridSearchCV", _fake_grid_search)
    features, target = split_features_target(feature_frame)

    train_model(features, target)

    assert captured["grid"] == tp.DEFAULT_PARAM_GRID


# --- metric generation -------------------------------------------


def test_evaluate_model_reports_expected_metrics(
    fitted_model: TransformedTargetRegressor, feature_frame: pd.DataFrame
) -> None:
    features, target = split_features_target(feature_frame)

    metrics = evaluate_model(fitted_model, features, target)

    assert set(metrics) == {"mape_pct", "rmse", "mae", "r2", "n_test_samples"}
    assert metrics["n_test_samples"] == len(target)
    assert metrics["mape_pct"] >= 0.0
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0
    assert metrics["r2"] <= 1.0
    assert metrics["r2"] > _MIN_LEARNABLE_R2


def test_evaluate_model_scores_perfect_predictions(feature_frame: pd.DataFrame) -> None:
    features, target = split_features_target(feature_frame)

    class _Oracle:
        def predict(self, _: pd.DataFrame) -> np.ndarray:
            return target.to_numpy(dtype=float)

    metrics = evaluate_model(_Oracle(), features, target)  # type: ignore[arg-type]

    assert metrics["mape_pct"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


# --- model storage ---------------------------------------------


def test_save_model_round_trips(
    tmp_path: Path, fitted_model: TransformedTargetRegressor, feature_frame: pd.DataFrame
) -> None:
    path = tmp_path / "06_models" / "model.joblib"
    features, _ = split_features_target(feature_frame)

    save_model(fitted_model, path)

    assert path.is_file()
    reloaded = load(path)
    np.testing.assert_allclose(reloaded.predict(features), fitted_model.predict(features))


def test_save_metrics_writes_sorted_json(tmp_path: Path) -> None:
    path = tmp_path / "08_reporting" / "training_metrics.json"
    metrics = {"mape_pct": 7.5, "rmse": 512.0, "mae": 310.0, "r2": 0.98, "n_test_samples": 42.0}

    save_metrics(metrics, path)

    assert path.is_file()
    assert pd.read_json(path, typ="series").to_dict() == metrics


# --- end to end -----------------------------------------------


def test_run_training_pipeline_writes_model_and_metrics(
    tmp_path: Path, feature_parquet: Path
) -> None:
    model_path = tmp_path / "06_models" / "diamantes.joblib"
    metrics_path = tmp_path / "08_reporting" / "metrics.json"

    model, metrics = run_training_pipeline(
        feature_parquet, model_path, metrics_path, param_grid=_NO_GRID, validate=False
    )

    assert model_path.is_file()
    assert metrics_path.is_file()
    assert set(metrics) == {"mape_pct", "rmse", "mae", "r2", "n_test_samples"}
    on_disk = pd.read_json(metrics_path, typ="series").to_dict()
    assert on_disk == pytest.approx(metrics)
    assert isinstance(load(model_path), TransformedTargetRegressor)
    assert np.isfinite(model.predict(pd.read_parquet(feature_parquet)[MODEL_FEATURES])).all()


def test_run_training_pipeline_runs_validation_steps(
    tmp_path: Path, feature_parquet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, int] = {}

    def _split_spy(
        train: tuple[pd.DataFrame, pd.Series], test: tuple[pd.DataFrame, pd.Series], **_: object
    ) -> None:
        seen["split_train"] = len(train[0])
        seen["split_test"] = len(test[0])

    def _model_spy(
        model: object,
        train: tuple[pd.DataFrame, pd.Series],
        test: tuple[pd.DataFrame, pd.Series],
        **_: object,
    ) -> dict[str, object]:
        seen["model_train"] = len(train[0])
        return {"diagnosis": {"verdict": "good_fit"}, "passed": True}

    monkeypatch.setattr(tp, "validate_train_test_split", _split_spy)
    monkeypatch.setattr(tp, "validate_model", _model_spy)

    run_training_pipeline(
        feature_parquet, tmp_path / "m.joblib", tmp_path / "metrics.json", param_grid=_NO_GRID
    )

    assert seen["split_train"] > seen["split_test"] > 0
    assert seen["model_train"] == seen["split_train"]


def test_run_training_pipeline_can_skip_validation(
    tmp_path: Path, feature_parquet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_: object, **__: object) -> None:
        raise AssertionError("validation should not run when validate=False")

    monkeypatch.setattr(tp, "validate_train_test_split", _boom)
    monkeypatch.setattr(tp, "validate_model", _boom)

    run_training_pipeline(
        feature_parquet,
        tmp_path / "m.joblib",
        tmp_path / "metrics.json",
        param_grid=_NO_GRID,
        validate=False,
    )


def test_main_invokes_pipeline_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(tp, "run_training_pipeline", lambda: calls.append("ran"))

    main()

    assert calls == ["ran"]


@pytest.mark.skipif(not FEATURE_PARQUET_PATH.is_file(), reason="feature table not built")
def test_pipeline_trains_on_real_feature_table() -> None:
    sample = load_feature_table().sample(n=3000, random_state=0)
    features, target = split_features_target(sample)
    x_train, x_test, y_train, y_test = split_train_test(features, target)

    model = train_model(x_train, y_train, param_grid=_NO_GRID)
    metrics = evaluate_model(model, x_test, y_test)

    assert metrics["mape_pct"] < _SMOKE_MAX_MAPE_PCT
    assert metrics["r2"] > _SMOKE_MIN_R2
