"""Unit tests for :mod:`pipelines.feature_pipeline.feature_pipeline` (issue #22)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.feature_pipeline import feature_pipeline as fp
from pipelines.feature_pipeline.feature_pipeline import (
    FEATURE_TABLE_COLUMNS,
    FEATURE_TABLE_DTYPES,
    MEASUREMENT_BOUNDS,
    PRICE_COLUMN,
    RAW_CSV_PATH,
    VOLUME_COLUMN,
    add_volume_feature,
    apply_domain_bounds,
    build_features,
    cast_feature_dtypes,
    coerce_numeric_columns,
    drop_invalid_rows,
    load_raw_data,
    main,
    restrict_categorical_columns,
    run_feature_pipeline,
    save_features,
)

RAW_COLUMNS = ["carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"]


def _valid_rows() -> list[dict[str, str]]:
    """Three clean rows that must survive the whole pipeline unchanged."""
    return [
        {
            "carat": "1.01",
            "cut": "Ideal",
            "color": "E",
            "clarity": "SI2",
            "depth": "61.5",
            "table": "55.0",
            "price": "3204",
            "x": "6.40",
            "y": "6.42",
            "z": "3.95",
        },
        {
            "carat": "0.90",
            "cut": "Premium",
            "color": "G",
            "clarity": "VS1",
            "depth": "62.0",
            "table": "58.0",
            "price": "4500",
            "x": "6.00",
            "y": "6.05",
            "z": "3.75",
        },
        {
            "carat": "2.15",
            "cut": "Very Good",
            "color": "H",
            "clarity": "VVS2",
            "depth": "60.1",
            "table": "57.0",
            "price": "15100",
            "x": "8.20",
            "y": "8.25",
            "z": "4.95",
        },
    ]


def _invalid_rows() -> list[dict[str, str]]:
    """One row per rejection reason handled by the pipeline."""
    base = _valid_rows()[0]
    return [
        {**base, "carat": "wetf"},  # non-numeric garbage -> NaN
        {**base, "color": "Z"},  # unknown colour grade -> NaN
        {**base, "clarity": "13243456"},  # numeric garbage in a category -> NaN
        {**base, "depth": "99.0"},  # above documented depth range -> NaN
        {**base, "x": "0.00"},  # impossible zero length -> NaN
        {**base, "y": "812.5"},  # far outside width range -> NaN
        {**base, "table": ""},  # missing value -> NaN
        dict(base),  # exact duplicate of a valid row
        {**base, "carat": "1.20", "z": "1.20"},  # z == carat column-swap error
        {**base, "depth": "55.0", "y": "55.0"},  # y == depth column-swap error
    ]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Raw-shaped frame (all text, like the CSV) mixing valid and invalid rows."""
    rows = _valid_rows() + _invalid_rows()
    return pd.DataFrame(rows, columns=RAW_COLUMNS).astype("object")


@pytest.fixture
def clean_numeric_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """``raw_df`` after numeric coercion + categorical + bounds validation."""
    return apply_domain_bounds(restrict_categorical_columns(coerce_numeric_columns(raw_df)))


# --- load_raw_data -------------------------------------------------------


def test_load_raw_data_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "does_not_exist.csv")


def test_load_raw_data_reads_without_coercion(tmp_path: Path, raw_df: pd.DataFrame) -> None:
    csv_path = tmp_path / "diamantes.csv"
    raw_df.to_csv(csv_path, index=False)

    loaded = load_raw_data(csv_path)

    assert list(loaded.columns) == RAW_COLUMNS
    assert (loaded.dtypes == "object").all()  # numbers stay as text until coerced


# --- coerce_numeric_columns -------------------------------------------


def test_coerce_numeric_columns_parses_numbers_and_nulls_garbage(raw_df: pd.DataFrame) -> None:
    result = coerce_numeric_columns(raw_df)

    assert pd.api.types.is_numeric_dtype(result["carat"])
    first_carat = 1.01
    assert result["carat"].iloc[0] == pytest.approx(first_carat)
    assert result["carat"].isna().any()  # the "wetf" row became NaN
    assert result["cut"].dtype == object  # categorical columns are left untouched


def test_coerce_numeric_columns_does_not_mutate_input(raw_df: pd.DataFrame) -> None:
    before = raw_df.copy()
    coerce_numeric_columns(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


# --- restrict_categorical_columns -----------------------------------


def test_restrict_categorical_columns_nulls_unknown_grades() -> None:
    df = pd.DataFrame({"cut": ["Ideal", "Bogus"], "color": ["E", "Z"], "clarity": ["SI2", "99"]})

    result = restrict_categorical_columns(df)

    assert result["cut"].tolist() == ["Ideal", None] or result["cut"].isna().iloc[1]
    assert result["color"].isna().tolist() == [False, True]
    assert result["clarity"].isna().tolist() == [False, True]


# --- apply_domain_bounds ---------------------------------------------


def test_apply_domain_bounds_nulls_out_of_range_and_zero_dimensions() -> None:
    df = pd.DataFrame(
        {
            "carat": [1.0, 9.9],  # 9.9 > 5.01
            "depth": [61.0, 61.0],
            "table": [57.0, 57.0],
            "x": [5.0, 0.0],  # 0.0 impossible
            "y": [5.0, 5.0],
            "z": [3.0, 3.0],
        }
    )

    result = apply_domain_bounds(df)

    assert np.isnan(result["carat"].iloc[1])
    assert np.isnan(result["x"].iloc[1])
    assert result["carat"].iloc[0] == pytest.approx(1.0)


def test_measurement_bounds_cover_every_dimension() -> None:
    for column in ("carat", "depth", "table", "x", "y", "z"):
        assert column in MEASUREMENT_BOUNDS


# --- drop_invalid_rows ----------------------------------------------


def test_drop_invalid_rows_removes_nan_duplicate_and_swapped(
    clean_numeric_df: pd.DataFrame,
) -> None:
    result = drop_invalid_rows(clean_numeric_df)

    expected_rows = len(_valid_rows())
    assert len(result) == expected_rows
    assert not result.isna().any().any()
    assert not result.duplicated().any()
    assert list(result.index) == list(range(expected_rows))


def test_drop_invalid_rows_keeps_column_swap_free_data() -> None:
    df = pd.DataFrame(
        {
            "carat": [1.0],
            "cut": ["Ideal"],
            "color": ["E"],
            "clarity": ["SI2"],
            "depth": [61.5],
            "table": [55.0],
            "price": [3204.0],
            "x": [6.4],
            "y": [6.42],
            "z": [3.95],
        }
    )

    assert len(drop_invalid_rows(df)) == 1


# --- add_volume_feature -------------------------------------------------


def test_add_volume_feature_multiplies_dimensions() -> None:
    df = pd.DataFrame({"x": [2.0, 3.0], "y": [3.0, 4.0], "z": [4.0, 5.0]})

    result = add_volume_feature(df)

    assert result[VOLUME_COLUMN].tolist() == pytest.approx([24.0, 60.0])
    assert "x" in result.columns  # source columns are kept


# --- cast_feature_dtypes ---------------------------------------------


def test_cast_feature_dtypes_matches_spec() -> None:
    df = pd.DataFrame(
        {
            "carat": [1.0],
            "cut": ["Ideal"],
            "color": ["E"],
            "clarity": ["SI2"],
            "depth": [61.5],
            "table": [55.0],
            "x": [6.4],
            "y": [6.42],
            "z": [3.95],
            VOLUME_COLUMN: [162.3],
            PRICE_COLUMN: [3204],
        }
    )

    result = cast_feature_dtypes(df)

    for column, dtype in FEATURE_TABLE_DTYPES.items():
        assert str(result[column].dtype) == dtype


# --- build_features (end to end, in memory) ---------------------------


def test_build_features_returns_only_clean_rows(raw_df: pd.DataFrame) -> None:
    features = build_features(raw_df)

    expected_rows = len(_valid_rows())
    assert list(features.columns) == FEATURE_TABLE_COLUMNS
    assert len(features) == expected_rows
    assert not features.isna().any().any()
    assert not features.duplicated().any()

    first = features.iloc[0]
    assert first["volume"] == pytest.approx(6.40 * 6.42 * 3.95, rel=1e-4)
    assert (features[PRICE_COLUMN] > 0).all()


def test_build_features_respects_documented_bounds(raw_df: pd.DataFrame) -> None:
    features = build_features(raw_df)

    for column, (low, high) in MEASUREMENT_BOUNDS.items():
        assert features[column].between(low, high).all()


# --- save_features / run_feature_pipeline ----------------------------


def test_save_features_round_trips(tmp_path: Path, raw_df: pd.DataFrame) -> None:
    features = build_features(raw_df)
    out_path = tmp_path / "nested" / "diamantes_features.parquet"

    save_features(features, out_path)

    assert out_path.is_file()
    pd.testing.assert_frame_equal(pd.read_parquet(out_path), features)


def test_run_feature_pipeline_writes_and_returns_feature_table(
    tmp_path: Path, raw_df: pd.DataFrame
) -> None:
    csv_path = tmp_path / "01_raw" / "diamantes.csv"
    csv_path.parent.mkdir(parents=True)
    raw_df.to_csv(csv_path, index=False)
    out_path = tmp_path / "04_feature" / "diamantes_features.parquet"

    returned = run_feature_pipeline(csv_path, out_path)

    assert out_path.is_file()
    pd.testing.assert_frame_equal(returned, build_features(load_raw_data(csv_path)))
    pd.testing.assert_frame_equal(pd.read_parquet(out_path), returned)


def test_main_invokes_pipeline_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(fp, "run_feature_pipeline", lambda: calls.append("ran"))

    main()

    assert calls == ["ran"]


# --- integration: the real course dataset ---------------------------


@pytest.mark.skipif(not RAW_CSV_PATH.is_file(), reason="raw diamantes.csv not present")
def test_pipeline_runs_on_real_dataset() -> None:
    features = build_features(load_raw_data(RAW_CSV_PATH))

    assert not features.empty
    assert list(features.columns) == FEATURE_TABLE_COLUMNS
    assert not features.isna().any().any()
    assert not features.duplicated().any()
    assert (features[PRICE_COLUMN] > 0).all()
    for column, (low, high) in MEASUREMENT_BOUNDS.items():
        assert features[column].between(low, high).all()
