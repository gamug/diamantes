import pandas as pd

from data.cleaning import (
    cast_clean_dtypes,
    classify_pattern,
    clean_column_values,
    clean_diamantes,
    get_all_masks,
    get_report_cleaning,
    get_report_masks,
)


def _sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "carat": ["1.01", "0.5x", "2.03"],  # row 1 is atypical
            "cut": ["Ideal", "Premium", "9999"],  # row 2 is atypical
            "color": ["G", "H", "E"],
            "clarity": ["VS2", "SI1", "SI2"],
            "depth": ["61.8", "62.0", "60.5"],
            "table": ["57.0", "58.0", "59.0"],
            "price": ["4268", "1500", "8500"],
            "x": [6.42, 7.32, 8.10],
            "y": [6.45, 7.28, 8.15],
            "z": [3.98, 4.53, 4.92],
        }
    )


def test_get_all_masks_replaces_letters_digits_and_whitespace() -> None:
    df = pd.DataFrame({"clarity": ["VS1"], "carat": ["1.02"]})
    masked = get_all_masks(df)
    assert masked.loc[0, "clarity"] == "LLD"
    assert masked.loc[0, "carat"] == "D.DD"


def test_classify_pattern() -> None:
    assert classify_pattern("NaN") == "Missing"
    assert classify_pattern(None) == "Missing"
    assert classify_pattern("D.DD") == "Numeric"
    assert classify_pattern("Lll") == "Alphabetic"
    assert classify_pattern("LLD") == "Alphanumeric"


def test_clean_column_values_flags_atypical_values_as_na() -> None:
    df = _sample_raw_df()
    cleaned = clean_column_values(df)

    assert pd.isna(cleaned.loc[1, "carat"])  # "0.5x" doesn't match the carat regex
    assert pd.isna(cleaned.loc[2, "cut"])  # "9999" doesn't match the cut regex
    # untouched columns/rows stay as-is
    assert cleaned.loc[0, "carat"] == "1.01"
    assert cleaned.loc[0, "cut"] == "Ideal"


def test_clean_column_values_does_not_mutate_input() -> None:
    df = _sample_raw_df()
    original = df.copy()
    clean_column_values(df)
    pd.testing.assert_frame_equal(df, original)


def test_cast_clean_dtypes() -> None:
    df = _sample_raw_df().drop(index=[1, 2]).reset_index(drop=True)
    cast = cast_clean_dtypes(df)
    assert cast["carat"].dtype == "float32"
    assert str(cast["cut"].dtype) == "category"
    assert cast["price"].dtype == "int64"


def test_clean_diamantes_drops_atypical_rows_and_casts_dtypes() -> None:
    df = _sample_raw_df()
    result = clean_diamantes(df)

    # rows 1 and 2 each had one atypical value -> dropped by dropna()
    assert len(result) == 1
    assert result["carat"].dtype == "float32"
    assert str(result["cut"].dtype) == "category"
    assert not result.isna().any().any()


def test_report_masks_and_cleaning_report_roundtrip() -> None:
    df = _sample_raw_df()
    mask_report = get_report_masks(df)

    # every row is accounted for, per column
    assert mask_report["carat"].sum() == len(df)

    report = get_report_cleaning(mask_report)
    assert "# Cleaning Report" in report
    assert "## Column: carat" in report
