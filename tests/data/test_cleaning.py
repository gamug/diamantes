import pandas as pd

from data.cleaning import (
    cast_clean_dtypes,
    classify_pattern,
    clean_column_values,
    clean_diamantes,
    get_all_masks,
    get_report_cleaning,
    get_report_masks,
    restrict_to_known_categories,
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


def test_get_all_masks_preserves_missing_values_as_nan_literal() -> None:
    # a real missing value must be reported as "NaN", not masked into "lll" (astype(str)
    # would otherwise stringify NaN to "nan" before the letter-masking pass runs)
    df = pd.DataFrame({"clarity": ["VS1", None], "carat": ["1.02", pd.NA]})
    masked = get_all_masks(df)
    assert masked.loc[1, "clarity"] == "NaN"
    assert masked.loc[1, "carat"] == "NaN"


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


def test_restrict_to_known_categories_flags_out_of_vocabulary_values() -> None:
    # "A" is a syntactically valid color (matches the raw regex, a single letter) but is not
    # one of the known GIA grades (D-J) in ORDINAL_CATEGORIES -- must still be caught
    df = pd.DataFrame({"color": ["G", "A"], "cut": ["Ideal", "Excellent"]})
    cleaned = restrict_to_known_categories(df)

    assert cleaned.loc[0, "color"] == "G"
    assert pd.isna(cleaned.loc[1, "color"])
    assert cleaned.loc[0, "cut"] == "Ideal"
    assert pd.isna(cleaned.loc[1, "cut"])  # "Excellent" is not a known cut grade


def test_clean_diamantes_drops_rows_with_unknown_but_syntactically_valid_category() -> None:
    df = _sample_raw_df().iloc[[0, 0]].reset_index(drop=True)  # two copies of the one valid row
    df.loc[1, "color"] = "A"  # passes the color regex (single letter) but isn't a GIA grade

    result = clean_diamantes(df)

    assert len(result) == 1
    assert result.iloc[0]["color"] == "G"


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
