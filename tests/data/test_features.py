import numpy as np
import pandas as pd

from data.features import (
    build_preprocessor,
    compute_volume,
    remove_corrupted_rows,
    remove_duplicate_rows,
    volume_feature_names_out,
)


def test_remove_duplicate_rows() -> None:
    df = pd.DataFrame({"carat": [1.0, 1.0, 2.0], "price": [100, 100, 200]})
    result = remove_duplicate_rows(df)
    assert len(result) == 2
    assert list(result.index) == [0, 1]


def test_remove_corrupted_rows_drops_column_swaps() -> None:
    df = pd.DataFrame(
        {
            "carat": [1.0, 2.0, 3.0],
            "depth": [61.5, 62.0, 5.0],  # row 2: y == depth (swap)
            "x": [6.4, 7.0, 8.0],
            "y": [6.5, 7.1, 5.0],
            "z": [1.0, 4.3, 5.0],  # row 0: z == carat (swap)
        }
    )
    result = remove_corrupted_rows(df)
    assert len(result) == 1
    assert result.iloc[0]["carat"] == 2.0


def test_compute_volume() -> None:
    x = np.array([[2.0, 3.0, 4.0], [1.0, 1.0, 1.0]])
    volume = compute_volume(x)
    assert volume.shape == (2, 1)
    np.testing.assert_allclose(volume.ravel(), [24.0, 1.0])


def test_volume_feature_names_out() -> None:
    names = volume_feature_names_out(None, [])  # type: ignore[arg-type]
    assert list(names) == ["volume"]


def test_build_preprocessor_fit_transform_shape() -> None:
    df = pd.DataFrame(
        {
            "carat": [1.0, 2.0, 1.5, np.nan],
            "depth": [61.5, 62.0, 61.0, 60.5],
            "table": [57.0, 58.0, 56.0, 57.5],
            "x": [6.4, 7.0, 6.5, 6.6],
            "y": [6.5, 7.1, 6.6, 6.7],
            "z": [4.0, 4.3, 4.1, 4.2],
            "cut": ["Ideal", "Premium", "Fair", "Good"],
            "color": ["G", "H", "D", "E"],
            "clarity": ["VS2", "SI1", "IF", "VVS1"],
        }
    )

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)

    # num (3) + volume (1) + cat_ordinal (3) = 7 output columns
    assert transformed.shape == (4, 7)
    feature_names = list(preprocessor.get_feature_names_out())
    assert "volume__volume" in feature_names
