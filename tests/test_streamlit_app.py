"""Unit tests for the :mod:`streamlit_app` online demo (issue #37).

The pure helpers are covered directly; the rendered page is exercised
end-to-end with Streamlit's headless ``AppTest`` (no browser).
"""

from __future__ import annotations

import io
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
    PREDICTION_COLUMN,
    RAW_INPUT_COLUMNS,
    SAMPLE_BATCH_CSV_PATH,
    batch_chart_data,
    batch_predict,
    batch_summary,
    build_input_row,
    fields_signature,
    input_warnings,
    load_sample_batch_csv,
    load_test_mape,
    missing_required_columns,
    predict_price,
    prepare_features,
    price_interval,
    range_warnings,
    read_batch_csv,
    submission_warnings,
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
    at.button(key="oor_go").click().run()  # confirm the out-of-range dialog

    assert len(at.exception) == 0
    html = _html(at)
    assert "dpx-notes" in html
    assert "extrapolation" in html


# --- online tab: out-of-range confirmation ----------------


def _oor_labels(at: AppTest) -> list[str]:
    return [button.label for button in at.button]


def test_range_warnings_flags_a_value_past_its_bound() -> None:
    ok = {**_BASE_FIELDS, "carat": 1.2}
    assert range_warnings(ok) == []

    flagged = range_warnings({**_BASE_FIELDS, "carat": 9.0, "depth": 40.0})
    assert any("Carat 9" in note and "0.2" in note for note in flagged)
    assert any("Depth % 40" in note for note in flagged)


def test_submission_warnings_merges_range_and_model_reliability_notes() -> None:
    notes = submission_warnings({**_BASE_FIELDS, "carat": 0.4})

    assert any("extrapolation" in note for note in notes)  # from input_warnings


def test_fields_signature_is_order_independent() -> None:
    a: dict[str, float | str] = {"carat": 1.0, "cut": "Ideal"}
    b: dict[str, float | str] = {"cut": "Ideal", "carat": 1.0}
    c: dict[str, float | str] = {"carat": 1.1, "cut": "Ideal"}

    assert fields_signature(a) == fields_signature(b)
    assert fields_signature(a) != fields_signature(c)


def test_online_tab_asks_before_pricing_out_of_range_input() -> None:
    at = AppTest.from_file(_APP, default_timeout=60).run()

    at.number_input[0].set_value(0.5).run()  # sub-1 ct -> outside the trained range
    at.button[0].click().run()

    assert len(at.exception) == 0
    assert "Estimate anyway" in _oor_labels(at)
    assert "Go back" in _oor_labels(at)
    assert "dpx-value__figure" not in _html(at)  # not priced yet


def test_online_tab_prices_after_the_user_confirms() -> None:
    at = AppTest.from_file(_APP, default_timeout=60).run()
    at.number_input[0].set_value(0.5).run()
    at.button[0].click().run()

    at.button(key="oor_go").click().run()

    assert len(at.exception) == 0
    assert "dpx-value__figure" in _html(at)


def test_online_tab_withholds_the_estimate_when_the_user_declines() -> None:
    at = AppTest.from_file(_APP, default_timeout=60).run()
    at.number_input[0].set_value(0.5).run()
    at.button[0].click().run()

    at.button(key="oor_back").click().run()

    assert len(at.exception) == 0
    assert "dpx-value__figure" not in _html(at)
    assert any("held back" in str(block.value) for block in at.markdown)


def test_online_tab_prices_in_range_input_without_a_dialog() -> None:
    at = AppTest.from_file(_APP, default_timeout=60).run()

    at.button[0].click().run()  # defaults are all in range

    assert len(at.exception) == 0
    assert "Estimate anyway" not in _oor_labels(at)
    assert "dpx-value__figure" in _html(at)


# --- batch inference (issue #38) --------------------------


#: The committed example batch file must stay a meaningful multi-row batch.
_MIN_SAMPLE_ROWS = 10


def _batch_frame(n: int = 4) -> pd.DataFrame:
    """``n`` identical valid raw rows, all cells text (like a CSV read as object)."""
    rows = [dict(_BASE_FIELDS) for _ in range(n)]
    return pd.DataFrame(rows, columns=RAW_INPUT_COLUMNS).astype(str)


def test_missing_required_columns_names_absent_columns_in_order() -> None:
    assert missing_required_columns(_batch_frame()) == []
    assert missing_required_columns(_batch_frame().drop(columns=["carat", "z"])) == ["carat", "z"]


def test_read_batch_csv_keeps_every_column_as_text() -> None:
    df = read_batch_csv(io.StringIO(_batch_frame().to_csv(index=False)))

    assert list(df.columns) == RAW_INPUT_COLUMNS
    assert all(pd.api.types.is_object_dtype(dtype) for dtype in df.dtypes)
    assert all(isinstance(value, str) for value in df.iloc[0])


def test_batch_predict_appends_one_prediction_per_row() -> None:
    n_rows = 5
    result = batch_predict(_DummyModel(), _batch_frame(n_rows))

    assert list(result.columns) == [*RAW_INPUT_COLUMNS, PREDICTION_COLUMN]
    assert len(result) == n_rows
    assert result[PREDICTION_COLUMN].dtype == float
    expected = 300.0 + 40.0 * (_X * _Y * _Z)
    assert result[PREDICTION_COLUMN].iloc[0] == pytest.approx(expected, rel=1e-4)


def test_batch_predict_preserves_row_order_and_carries_extra_columns() -> None:
    frame = _batch_frame(3)
    frame["price"] = ["1", "2", "3"]
    frame.index = [10, 20, 30]

    result = batch_predict(_DummyModel(), frame)

    assert list(result["price"]) == ["1", "2", "3"]
    assert list(result.index) == [0, 1, 2]
    assert PREDICTION_COLUMN in result.columns


def test_batch_predict_rejects_a_file_missing_a_column() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        batch_predict(_DummyModel(), _batch_frame().drop(columns=["depth"]))


def test_batch_predict_rejects_a_header_only_file() -> None:
    with pytest.raises(ValueError, match="no data rows"):
        batch_predict(_DummyModel(), _batch_frame(0))


def test_batch_predict_rejects_a_reserved_prediction_column() -> None:
    frame = _batch_frame(2)
    frame[PREDICTION_COLUMN] = ["1", "2"]

    with pytest.raises(ValueError, match="reserved output name"):
        batch_predict(_DummyModel(), frame)


def test_batch_predict_rejects_a_wholly_unusable_feature_column() -> None:
    frame = _batch_frame(3)
    frame["carat"] = "not-a-number"  # every row's carat is junk -> feature all-NaN

    with pytest.raises(ValueError, match="no usable value"):
        batch_predict(_DummyModel(), frame)


def test_batch_predict_still_scores_rows_with_unparseable_cells() -> None:
    n_rows = 2
    frame = _batch_frame(n_rows)
    frame.loc[0, "carat"] = "not-a-number"
    frame.loc[1, "clarity"] = "ZZ"

    result = batch_predict(_DummyModel(), frame)

    assert len(result) == n_rows  # row-preserving: nothing dropped, imputers handle the gaps


def test_batch_predict_with_a_real_pipeline_prices_every_row() -> None:
    rng = np.random.default_rng(1)
    n = 120
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

    batch_rows = 6
    result = batch_predict(model, _batch_frame(batch_rows))

    assert len(result) == batch_rows
    assert (result[PREDICTION_COLUMN] > 0.0).all()


def test_batch_summary_reports_price_stats() -> None:
    result = pd.DataFrame({PREDICTION_COLUMN: [100.0, 200.0, 300.0]})

    assert batch_summary(result) == pytest.approx({"mean": 200.0, "min": 100.0, "max": 300.0})


def test_batch_chart_data_is_numeric_and_drops_nans() -> None:
    result = pd.DataFrame({"carat": ["1.0", "bad", "2.0"], PREDICTION_COLUMN: [10.0, 20.0, 30.0]})

    chart = batch_chart_data(result)

    numeric_rows = 2  # the "bad" carat row is dropped
    assert list(chart.columns) == ["carat", PREDICTION_COLUMN]
    assert len(chart) == numeric_rows
    assert chart["carat"].dtype == float


def test_committed_sample_batch_csv_round_trips_through_the_transform() -> None:
    assert SAMPLE_BATCH_CSV_PATH.is_file()
    text = load_sample_batch_csv()
    assert text is not None
    assert text.splitlines()[0] == ",".join(RAW_INPUT_COLUMNS)

    df = read_batch_csv(io.StringIO(text))
    assert missing_required_columns(df) == []
    assert len(df) >= _MIN_SAMPLE_ROWS
    assert len(batch_predict(_DummyModel(), df)) == len(df)


def test_load_sample_batch_csv_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_sample_batch_csv(tmp_path / "nope.csv") is None


def test_page_has_an_online_tab_and_a_batch_tab() -> None:
    at = AppTest.from_file(_APP, default_timeout=60).run()

    assert len(at.exception) == 0
    assert [tab.label for tab in at.tabs] == ["One stone", "A file of stones"]
    assert len(at.file_uploader) == 1


def test_batch_tab_prices_an_uploaded_file() -> None:
    at = AppTest.from_file(_APP, default_timeout=60).run()
    sample = load_sample_batch_csv()
    assert sample is not None

    at.file_uploader[0].upload("diamantes_batch_sample.csv", sample.encode(), "text/csv").run()

    assert len(at.exception) == 0
    assert any(PREDICTION_COLUMN in list(getattr(df.value, "columns", [])) for df in at.dataframe)
    assert "Download predictions (CSV)" in [button.label for button in at.download_button]


def test_batch_tab_reports_a_file_that_is_missing_columns() -> None:
    at = AppTest.from_file(_APP, default_timeout=60).run()

    at.file_uploader[0].upload("bad.csv", b"carat,cut\n1.0,Ideal\n", "text/csv").run()

    assert len(at.exception) == 0
    assert any("missing required columns" in str(error.value) for error in at.error)


def test_batch_tab_reports_a_header_only_file_without_crashing() -> None:
    at = AppTest.from_file(_APP, default_timeout=60).run()
    header_only = (",".join(RAW_INPUT_COLUMNS) + "\n").encode()

    at.file_uploader[0].upload("empty.csv", header_only, "text/csv").run()

    assert len(at.exception) == 0
    assert any("no data rows" in str(error.value) for error in at.error)
