"""Unit tests for the :mod:`inference_pipeline` script (issue #27).

Uses a dummy model plus synthetic raw data; one test also feeds the prepared
features through a real (synthetically fitted) sklearn pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import dump

import inference_pipeline as ip
from diamond_features import KNOWN_CATEGORIES
from inference_pipeline import (
    PREDICTION_COLUMN,
    build_predictions_frame,
    load_model,
    load_new_data,
    plot_predictions,
    predict_prices,
    prepare_features,
    run_inference_pipeline,
    save_predictions,
)
from training_pipeline import MODEL_FEATURES, build_model_pipeline

RAW_COLUMNS = ["carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"]


class DummyModel:
    """A picklable stand-in: price is a linear function of carat and volume."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        carat = pd.to_numeric(features["carat"], errors="coerce").fillna(0.0)
        volume = pd.to_numeric(features["volume"], errors="coerce").fillna(0.0)
        return np.asarray(300.0 + 2500.0 * carat + 5.0 * volume, dtype=float)


def _clean_rows() -> list[dict[str, str]]:
    return [
        {
            "carat": "1.01",
            "cut": "Ideal",
            "color": "E",
            "clarity": "SI2",
            "depth": "61.5",
            "table": "55.0",
            "price": "3200",
            "x": "6.40",
            "y": "6.42",
            "z": "3.95",
        },
        {
            "carat": "0.70",
            "cut": "Premium",
            "color": "G",
            "clarity": "VS1",
            "depth": "62.0",
            "table": "58.0",
            "price": "2500",
            "x": "5.70",
            "y": "5.72",
            "z": "3.54",
        },
        {
            "carat": "1.52",
            "cut": "Very Good",
            "color": "H",
            "clarity": "VVS2",
            "depth": "60.1",
            "table": "57.0",
            "price": "11000",
            "x": "7.35",
            "y": "7.40",
            "z": "4.44",
        },
    ]


def _messy_rows() -> list[dict[str, str]]:
    base = _clean_rows()[0]
    return [
        {**base, "carat": "wetf"},  # non-numeric -> NaN
        {**base, "clarity": "ZZZ"},  # unknown grade -> NaN
        {**base, "x": ""},  # missing dimension -> NaN volume
        {**base, "depth": "300.0"},  # out of range -> NaN
    ]


def _raw_frame(*, messy: bool = True) -> pd.DataFrame:
    rows = _clean_rows() + (_messy_rows() if messy else [])
    return pd.DataFrame(rows, columns=RAW_COLUMNS).astype("object")


@pytest.fixture
def dummy_model_path(tmp_path: Path) -> Path:
    path = tmp_path / "06_models" / "dummy.joblib"
    path.parent.mkdir(parents=True)
    dump(DummyModel(), path)
    return path


# --- model loading ------------------------------------------------


def test_load_model_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        load_model(tmp_path / "nope.joblib")


def test_load_model_round_trips(dummy_model_path: Path) -> None:
    model = load_model(dummy_model_path)

    features = pd.DataFrame({"carat": [1.0], "volume": [100.0]})
    assert model.predict(features)[0] == pytest.approx(300.0 + 2500.0 + 500.0)


def test_load_new_data_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_new_data(tmp_path / "missing.csv")


def test_load_new_data_reads_columns_as_text(tmp_path: Path) -> None:
    csv_path = tmp_path / "new.csv"
    _raw_frame().to_csv(csv_path, index=False)

    loaded = load_new_data(csv_path)

    assert list(loaded.columns) == RAW_COLUMNS
    assert (loaded.dtypes == "object").all()


# --- transformation --------------------------------------------


def test_prepare_features_keeps_every_row_and_selects_model_columns() -> None:
    raw = _raw_frame()

    features = prepare_features(raw)

    assert list(features.columns) == MODEL_FEATURES
    assert len(features) == len(raw)  # no rows dropped at inference time
    assert features.isna().any().any()  # the messy rows leave NaNs for the imputer


def test_prepare_features_rejects_missing_required_columns() -> None:
    raw = _raw_frame().drop(columns=["carat"])

    with pytest.raises(ValueError, match="missing required columns"):
        prepare_features(raw)


def test_prepare_features_feed_a_real_sklearn_pipeline() -> None:
    rng = np.random.default_rng(0)
    n = 200
    train = pd.DataFrame(
        {
            "carat": rng.uniform(0.3, 2.5, n).astype("float32"),
            "depth": rng.uniform(58.0, 65.0, n).astype("float32"),
            "table": rng.uniform(54.0, 60.0, n).astype("float32"),
            "volume": rng.uniform(30.0, 400.0, n).astype("float32"),
            "cut": pd.Categorical(rng.choice(KNOWN_CATEGORIES["cut"], n)),
            "color": pd.Categorical(rng.choice(KNOWN_CATEGORIES["color"], n)),
            "clarity": pd.Categorical(rng.choice(KNOWN_CATEGORIES["clarity"], n)),
        }
    )
    y = 3000.0 * train["carat"].astype(float) + 10.0 * train["volume"].astype(float)
    model = build_model_pipeline().fit(train[MODEL_FEATURES], y)

    predictions = predict_prices(model, prepare_features(_raw_frame()))

    assert predictions.shape == (len(_raw_frame()),)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


# --- prediction assembly / storage --------------------------


def test_predict_prices_returns_a_float_array(dummy_model_path: Path) -> None:
    model = load_model(dummy_model_path)
    features = prepare_features(_raw_frame())

    predictions = predict_prices(model, features)

    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype == float
    assert len(predictions) == len(features)


def test_build_predictions_frame_appends_prediction_column() -> None:
    raw = _raw_frame()
    predictions = np.arange(len(raw), dtype=float)

    frame = build_predictions_frame(raw, predictions)

    assert list(frame.columns) == [*RAW_COLUMNS, PREDICTION_COLUMN]
    assert frame[PREDICTION_COLUMN].tolist() == predictions.tolist()
    assert list(frame.index) == list(range(len(raw)))


def test_save_predictions_round_trips(tmp_path: Path) -> None:
    frame = build_predictions_frame(_raw_frame(), np.zeros(len(_raw_frame())))
    path = tmp_path / "nested" / "predictions.parquet"

    save_predictions(frame, path)

    assert path.is_file()
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)


def test_plot_predictions_writes_a_png(tmp_path: Path) -> None:
    frame = build_predictions_frame(_raw_frame(), np.linspace(300, 15000, len(_raw_frame())))
    path = tmp_path / "plots" / "predictions.png"

    plot_predictions(frame, path)

    assert path.is_file()
    assert path.read_bytes().startswith(b"\x89PNG")


# --- end to end ---------------------------------------------


def test_run_inference_pipeline_end_to_end(tmp_path: Path, dummy_model_path: Path) -> None:
    csv_path = tmp_path / "01_raw" / "new.csv"
    csv_path.parent.mkdir(parents=True)
    raw = _raw_frame()
    raw.to_csv(csv_path, index=False)
    out_path = tmp_path / "07_model_output" / "predictions.parquet"
    plot_path = tmp_path / "07_model_output" / "predictions.png"

    result = run_inference_pipeline(csv_path, dummy_model_path, out_path, plot_path)

    assert len(result) == len(raw)  # every input row scored, none dropped
    assert out_path.is_file()
    assert plot_path.is_file()

    on_disk = pd.read_parquet(out_path)
    assert list(on_disk.columns) == [*RAW_COLUMNS, PREDICTION_COLUMN]
    np.testing.assert_allclose(
        on_disk[PREDICTION_COLUMN].to_numpy(), result[PREDICTION_COLUMN].to_numpy()
    )

    expected = DummyModel().predict(prepare_features(raw))
    np.testing.assert_allclose(result[PREDICTION_COLUMN].to_numpy(), expected)
    assert np.isfinite(result[PREDICTION_COLUMN].to_numpy()).all()  # messy rows included


def test_run_inference_pipeline_uses_default_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ip, "run_inference_pipeline", lambda: calls.append("ran"))

    ip.main()

    assert calls == ["ran"]
