"""Unit tests for the :mod:`feature_pipeline` script (issue #22)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import feature_pipeline as fp
from diamond_features import FEATURE_TABLE_COLUMNS, MEASUREMENT_BOUNDS, PRICE_COLUMN
from feature_pipeline import (
    RAW_CSV_PATH,
    build_features,
    load_raw_data,
    main,
    run_feature_pipeline,
    save_features,
)
from feature_validation import FeatureValidationError, validate_features

RAW_COLUMNS = ["carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A small raw-shaped frame (all text) with two clean rows and two rejects."""
    rows = [
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
            "carat": "wetf",
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
            "carat": "1.10",
            "cut": "Ideal",
            "color": "Q",
            "clarity": "SI2",
            "depth": "61.5",
            "table": "55.0",
            "price": "3300",
            "x": "6.40",
            "y": "6.42",
            "z": "3.95",
        },
    ]
    return pd.DataFrame(rows, columns=RAW_COLUMNS).astype("object")


def test_load_raw_data_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "does_not_exist.csv")


def test_load_raw_data_reads_without_coercion(tmp_path: Path, raw_df: pd.DataFrame) -> None:
    csv_path = tmp_path / "diamantes.csv"
    raw_df.to_csv(csv_path, index=False)

    loaded = load_raw_data(csv_path)

    assert list(loaded.columns) == RAW_COLUMNS
    assert (loaded.dtypes == "object").all()  # numbers stay text until transformed


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

    expected_clean_rows = 2
    assert len(returned) == expected_clean_rows
    assert list(returned.columns) == FEATURE_TABLE_COLUMNS
    assert not returned.isna().any().any()
    assert out_path.is_file()
    pd.testing.assert_frame_equal(pd.read_parquet(out_path), returned)


def test_run_feature_pipeline_does_not_persist_when_validation_fails(
    tmp_path: Path, raw_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "01_raw" / "diamantes.csv"
    csv_path.parent.mkdir(parents=True)
    raw_df.to_csv(csv_path, index=False)
    out_path = tmp_path / "04_feature" / "diamantes_features.parquet"

    # build_features returns a table that breaks a range rule -> validation must gate it
    def _broken_build(_: pd.DataFrame) -> pd.DataFrame:
        good = build_features(raw_df)
        good.loc[good.index[0], "carat"] = pd.NA  # forces a null-fraction failure
        return good

    monkeypatch.setattr(fp, "build_features", _broken_build)

    with pytest.raises(FeatureValidationError):
        run_feature_pipeline(csv_path, out_path)

    assert not out_path.exists()  # nothing persisted on a failed validation


def test_main_invokes_pipeline_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(fp, "run_feature_pipeline", lambda: calls.append("ran"))

    main()

    assert calls == ["ran"]


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

    # the real course dataset must satisfy the full validation contract
    pd.testing.assert_frame_equal(validate_features(features), features)
