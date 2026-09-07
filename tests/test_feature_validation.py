"""Unit tests for :mod:`feature_validation` with valid and invalid data (issue #23)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from diamond_features import (
    FEATURE_TABLE_COLUMNS,
    PRICE_COLUMN,
    VOLUME_COLUMN,
    cast_feature_dtypes,
)
from feature_validation import (
    FeatureValidationError,
    build_feature_schema,
    validate_features,
)


def _valid_feature_table() -> pd.DataFrame:
    """Three internally consistent, in-range, unique feature rows.

    ``depth`` matches ``2*z / (x + y) * 100`` and ``volume`` matches
    ``x * y * z`` for every row, so the integrity checks pass.
    """
    rows = [
        # x=6.0, y=6.0, z=3.6 -> depth 60.0 ; volume 129.6
        {
            "carat": 1.00,
            "cut": "Ideal",
            "color": "E",
            "clarity": "SI2",
            "depth": 60.0,
            "table": 57.0,
            "x": 6.0,
            "y": 6.0,
            "z": 3.6,
            VOLUME_COLUMN: 6.0 * 6.0 * 3.6,
            PRICE_COLUMN: 5000,
        },
        # x=6.5, y=6.5, z=4.0 -> depth ~61.538 ; volume 169.0
        {
            "carat": 1.20,
            "cut": "Premium",
            "color": "G",
            "clarity": "VS1",
            "depth": 2 * 4.0 / 13.0 * 100,
            "table": 58.0,
            "x": 6.5,
            "y": 6.5,
            "z": 4.0,
            VOLUME_COLUMN: 6.5 * 6.5 * 4.0,
            PRICE_COLUMN: 8000,
        },
        # x=5.0, y=5.0, z=3.1 -> depth 62.0 ; volume 77.5
        {
            "carat": 0.50,
            "cut": "Good",
            "color": "H",
            "clarity": "SI1",
            "depth": 62.0,
            "table": 59.0,
            "x": 5.0,
            "y": 5.0,
            "z": 3.1,
            VOLUME_COLUMN: 5.0 * 5.0 * 3.1,
            PRICE_COLUMN: 1500,
        },
    ]
    df = pd.DataFrame(rows)[FEATURE_TABLE_COLUMNS]
    return cast_feature_dtypes(df)


@pytest.fixture
def valid_features() -> pd.DataFrame:
    return _valid_feature_table()


# --- valid data --------------------------------------------------------


def test_validate_features_accepts_a_clean_table(valid_features: pd.DataFrame) -> None:
    result = validate_features(valid_features)

    pd.testing.assert_frame_equal(result, valid_features)


def test_build_feature_schema_declares_columns_in_table_order() -> None:
    schema = build_feature_schema()

    assert list(schema.columns) == FEATURE_TABLE_COLUMNS
    assert schema.strict is True
    assert schema.ordered is True


# --- invalid data ----------------------------------------------------


def test_rejects_measurement_out_of_range(valid_features: pd.DataFrame) -> None:
    bad = valid_features.copy()
    bad.loc[0, "carat"] = np.float32(10.0)  # > documented max 5.01

    with pytest.raises(FeatureValidationError, match="in_range"):
        validate_features(bad)


def test_rejects_unknown_category(valid_features: pd.DataFrame) -> None:
    bad = valid_features.copy()
    bad["color"] = bad["color"].cat.add_categories(["Q"])
    bad.loc[0, "color"] = "Q"

    with pytest.raises(FeatureValidationError, match="isin"):
        validate_features(bad)


def test_rejects_wrong_dtype(valid_features: pd.DataFrame) -> None:
    bad = valid_features.copy()
    bad[PRICE_COLUMN] = bad[PRICE_COLUMN].astype("float64")

    with pytest.raises(FeatureValidationError):
        validate_features(bad)


def test_rejects_negative_price(valid_features: pd.DataFrame) -> None:
    bad = valid_features.copy()
    bad.loc[0, PRICE_COLUMN] = -1

    with pytest.raises(FeatureValidationError, match="greater_than"):
        validate_features(bad)


def test_rejects_nulls_with_zero_tolerance(valid_features: pd.DataFrame) -> None:
    bad = valid_features.copy()
    bad.loc[0, "table"] = np.nan

    with pytest.raises(FeatureValidationError, match="null fraction"):
        validate_features(bad)


def test_allows_nulls_below_threshold(valid_features: pd.DataFrame) -> None:
    sparse = valid_features.copy()
    sparse.loc[0, "table"] = np.nan  # 1 of 3 rows == 33%

    result = validate_features(sparse, max_null_fraction=0.4)

    assert len(result) == len(valid_features)


def test_rejects_null_fraction_over_threshold(valid_features: pd.DataFrame) -> None:
    bad = valid_features.copy()
    bad.loc[0, "table"] = np.nan  # 33% > 20%

    with pytest.raises(FeatureValidationError, match="null fraction"):
        validate_features(bad, max_null_fraction=0.2)


def test_rejects_duplicate_rows(valid_features: pd.DataFrame) -> None:
    bad = pd.concat([valid_features, valid_features.iloc[[0]]], ignore_index=True)

    with pytest.raises(FeatureValidationError, match="unique"):
        validate_features(bad)


def test_rejects_volume_not_matching_dimensions(valid_features: pd.DataFrame) -> None:
    bad = valid_features.copy()
    bad.loc[0, VOLUME_COLUMN] = np.float32(999.0)

    with pytest.raises(FeatureValidationError, match="volume_equals_x_times_y_times_z"):
        validate_features(bad)


def test_rejects_z_equals_carat_column_swap(valid_features: pd.DataFrame) -> None:
    bad = valid_features.copy()
    # x=6, y=6, z=3.6 keeps depth/volume consistent; carat == z triggers only the swap check
    bad.loc[0, "carat"] = np.float32(3.6)

    with pytest.raises(FeatureValidationError, match="no_z_carat_column_swap"):
        validate_features(bad)


def test_rejects_depth_inconsistent_with_geometry(valid_features: pd.DataFrame) -> None:
    bad = valid_features.copy()
    bad.loc[0, "depth"] = np.float32(70.0)  # in range, but geometry implies 60.0

    with pytest.raises(FeatureValidationError, match="depth_matches_geometry"):
        validate_features(bad)


def test_error_message_enumerates_every_failing_rule(valid_features: pd.DataFrame) -> None:
    bad = valid_features.copy()
    bad.loc[0, "carat"] = np.float32(10.0)
    bad["color"] = bad["color"].cat.add_categories(["Q"])
    bad.loc[1, "color"] = "Q"

    with pytest.raises(FeatureValidationError) as excinfo:
        validate_features(bad)

    message = str(excinfo.value)
    assert "in_range" in message
    assert "isin" in message
