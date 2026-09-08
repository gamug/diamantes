"""Unit tests for the :mod:`streamlit_app` online demo (issue #37).

The pure helpers are covered directly; the rendered page is exercised
end-to-end with Streamlit's headless ``AppTest`` (no browser).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from diamond_features import KNOWN_CATEGORIES, MEASUREMENT_BOUNDS
from streamlit_app import (
    DEFAULT_TEST_MAPE_PCT,
    MODEL_FEATURES,
    RAW_INPUT_COLUMNS,
    build_input_row,
    input_warnings,
    load_test_mape,
    predict_price,
    prepare_features,
    price_interval,
)
from training_pipeline import build_model_pipeline

_X, _Y, _Z = 6.0, 6.0, 3.7
_BASE_FIELDS: dict[str, float | str] = {
    "carat": 1.2,
    "cut": "Ideal",
    "color": "G",
    "clarity": "VS2",
    "depth": 61.7,
    "table": 57.0,
    "x": _X,
    "y": _Y,
    "z": _Z,
}


def _row(**overrides: float | str) -> pd.DataFrame:
    return build_input_row({**_BASE_FIELDS, **overrides})


class _DummyModel:
    """Price is a linear function of the engineered ``volume`` column."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return 300.0 + 40.0 * features["volume"].to_numpy(dtype=float)


# --- constants stay aligned with the rest of the codebase ----


def test_module_constants_match_diamond_features() -> None:
    assert set(MODEL_FEATURES) - {"volume"} <= set(RAW_INPUT_COLUMNS)
    assert {"cut", "color", "clarity"} == set(KNOWN_CATEGORIES)
    numeric_fields = [c for c in RAW_INPUT_COLUMNS if c not in KNOWN_CATEGORIES]
    assert all(field in MEASUREMENT_BOUNDS for field in numeric_fields)


# --- build_input_row / prepare_features -------------------


def test_build_input_row_is_one_text_row() -> None:
    row = _row()

    assert list(row.columns) == RAW_INPUT_COLUMNS
    assert len(row) == 1
    assert all(isinstance(value, str) for value in row.iloc[0])


def test_build_input_row_rejects_a_missing_field() -> None:
    partial = {key: value for key, value in _BASE_FIELDS.items() if key != "carat"}

    with pytest.raises(ValueError, match="missing or empty input fields"):
        build_input_row(partial)


def test_build_input_row_rejects_a_blank_field() -> None:
    with pytest.raises(ValueError, match="missing or empty input fields"):
        build_input_row({**_BASE_FIELDS, "clarity": "  "})


def test_prepare_features_returns_clean_model_columns() -> None:
    features = prepare_features(_row())

    assert list(features.columns) == MODEL_FEATURES
    assert len(features) == 1
    assert not features.isna().any().any()
    assert features["volume"].iloc[0] == pytest.approx(_X * _Y * _Z, rel=1e-4)


# --- predict_price ----------------------------------------


def test_predict_price_runs_the_full_transform_then_the_model() -> None:
    row = _row()

    price = predict_price(_DummyModel(), row)

    expected = 300.0 + 40.0 * (_X * _Y * _Z)
    assert price == pytest.approx(expected, rel=1e-4)


def test_predict_price_with_a_real_pipeline_is_a_positive_float() -> None:
    rng = np.random.default_rng(0)
    n = 150
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
    target = 3000.0 * train["carat"].astype(float) + 10.0 * train["volume"].astype(float)
    model = build_model_pipeline().fit(train[MODEL_FEATURES], target)

    price = predict_price(model, _row())

    assert isinstance(price, float)
    assert price > 0.0


# --- load_test_mape --------------------------------------


def test_load_test_mape_reads_the_metrics_file(tmp_path: Path) -> None:
    path = tmp_path / "training_metrics.json"
    path.write_text(json.dumps({"mape_pct": 5.5, "r2": 0.98}))

    assert load_test_mape(path) == pytest.approx(5.5)


def test_load_test_mape_falls_back_when_file_missing(tmp_path: Path) -> None:
    assert load_test_mape(tmp_path / "nope.json") == DEFAULT_TEST_MAPE_PCT


def test_load_test_mape_falls_back_on_unusable_json(tmp_path: Path) -> None:
    path = tmp_path / "training_metrics.json"
    path.write_text("{}")

    assert load_test_mape(path) == DEFAULT_TEST_MAPE_PCT


# --- price_interval -------------------------------------


def test_price_interval_is_symmetric_percentage_band() -> None:
    assert price_interval(1000.0, 10.0) == pytest.approx((900.0, 1100.0))
    assert price_interval(2000.0, 0.0) == pytest.approx((2000.0, 2000.0))


# --- input_warnings ------------------------------------


def test_input_warnings_flags_sub_carat_extrapolation() -> None:
    warnings = input_warnings(carat=0.5, depth=61.7, x=6.0, y=6.0, z=3.7)

    assert len(warnings) == 1
    assert "extrapolation" in warnings[0]


def test_input_warnings_flags_geometry_mismatch() -> None:
    warnings = input_warnings(carat=1.5, depth=75.0, x=6.0, y=6.0, z=3.7)

    assert any("doesn't match what x, y and z imply" in message for message in warnings)


def test_input_warnings_clean_for_consistent_input() -> None:
    assert input_warnings(carat=1.5, depth=61.7, x=6.0, y=6.02, z=3.71) == []


def test_input_warnings_skips_geometry_when_a_dimension_is_zero() -> None:
    warnings = input_warnings(carat=1.5, depth=90.0, x=0.0, y=6.0, z=3.7)

    assert warnings == []


# --- the rendered page (headless AppTest) ------------------


_APP = str(Path(__file__).parents[1] / "src" / "streamlit_app.py")
_N_NUMBER_FIELDS = len(RAW_INPUT_COLUMNS) - len(KNOWN_CATEGORIES)  # carat/depth/table/x/y/z
_N_SELECT_FIELDS = len(KNOWN_CATEGORIES)  # cut/color/clarity


def _html(at: AppTest) -> str:
    return " ".join(str(getattr(block, "body", "")) for block in at.get("html"))


def test_page_renders_the_form_without_error() -> None:
    at = AppTest.from_file(_APP, default_timeout=60).run()

    assert len(at.exception) == 0
    assert len(at.number_input) == _N_NUMBER_FIELDS
    assert len(at.selectbox) == _N_SELECT_FIELDS
    assert "dpx-hero" in _html(at)


def test_page_shows_an_estimate_on_submit() -> None:
    at = AppTest.from_file(_APP, default_timeout=60).run()

    at.button[0].click().run()

    assert len(at.exception) == 0
    html = _html(at)
    assert "dpx-value__figure" in html
    assert "Estimated market value" in html
    assert len(at.dataframe) == 1  # "What the model used" only renders with an estimate


def test_page_surfaces_a_note_for_a_sub_carat_stone() -> None:
    at = AppTest.from_file(_APP, default_timeout=60).run()

    at.number_input[0].set_value(0.5).run()  # carat is the first number input
    at.button[0].click().run()

    assert len(at.exception) == 0
    html = _html(at)
    assert "dpx-notes" in html
    assert "extrapolation" in html
