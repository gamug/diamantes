"""Raw-data cleaning: detect and drop atypical/corrupted values.

Ported from ``notebooks/3-analysis/02-jrz-data_description_Manual-pandas-2024_10_24.ipynb``.
The approach: mask every value down to its character "shape" (``L``etter,
``l``owercase, ``D``igit, ``s``pace), report which shapes are prevalent vs.
rare per column, then replace any value that doesn't match the expected
per-column regex with ``NA``.
"""

import pandas as pd

from data.constants import CLEAN_DTYPES, ORDINAL_CATEGORIES, RAW_COLUMN_REGEX

#: Character-class replacements used to turn a raw value into its "shape"
#: mask, e.g. ``"VS1"`` -> ``"LLD"``, ``"1.02"`` -> ``"D.DD"``.
_MASK_REPLACEMENTS: dict[str, str] = {
    "[A-Z]": "L",
    "[a-z]": "l",
    "[0-9]": "D",
    r"\s": "s",
}


def get_all_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Replace every value in ``df`` with its character-shape mask.

    Args:
        df: The input DataFrame to be masked.

    Returns:
        A same-shaped DataFrame of strings where letters, digits and
        whitespace have been replaced by ``L``/``l``/``D``/``s`` respectively
        (e.g. ``"VS1"`` -> ``"LLD"``), and missing values are preserved as
        the literal string ``"NaN"`` (not masked into ``"lll"``).
    """
    # nullable "string" dtype (not plain `str`) so missing values stay NA through the
    # replacements below, instead of being stringified to "nan" and then masked into "lll"
    df_string = df.astype("string")
    for pattern, replacement in _MASK_REPLACEMENTS.items():
        df_string = pd.concat(
            [
                df_string[col].str.replace(pattern, replacement, regex=True)
                for col in df_string.columns
            ],
            axis=1,
        )
    return df_string.fillna("NaN")


def classify_pattern(pattern: str | None) -> str:
    """Classify a shape-mask pattern as numeric, alphabetic, alphanumeric or missing.

    Args:
        pattern: A shape-mask value produced by :func:`get_all_masks`
            (e.g. ``"DD.DD"``, ``"Llll"``), or ``"NaN"``/``None`` for missing.

    Returns:
        One of ``"Missing"``, ``"Numeric"``, ``"Alphabetic"``,
        ``"Alphanumeric"`` or ``"Unknown"``.
    """
    if pattern is None or pattern == "NaN":
        return "Missing"
    if all(ch in ("D", ".", "-") for ch in pattern):
        return "Numeric"
    if all(ch == "L" or ch.islower() for ch in pattern):
        return "Alphabetic"
    if "L" in pattern and "D" in pattern:
        return "Alphanumeric"
    return "Unknown"


def get_report_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many rows fall into each shape-mask pattern, per column.

    Args:
        df: The input DataFrame to generate the report from.

    Returns:
        A DataFrame indexed by shape-mask pattern (rows) x original column
        (columns), with the count of values matching each pattern. Missing
        values are reported under the ``"NaN"`` index.
    """
    masked_df = get_all_masks(df)
    report = {col: masked_df[col].value_counts(dropna=False) for col in masked_df.columns}
    report_df = pd.DataFrame(report)
    index = report_df.index.to_series()
    index.loc[index.isna()] = "NaN"
    report_df.index = pd.Index(index)
    return report_df.fillna(0).astype(int)


def get_report_cleaning(mask_report: pd.DataFrame, atypic_threshold: float = 0.05) -> str:
    """Render a human-readable markdown cleaning report from a mask report.

    For each column, shows the prevalent (typical) value shape, its inferred
    classification, any rare/atypical shapes (below ``atypic_threshold`` of
    the column's rows), and the share of missing values.

    Args:
        mask_report: The report produced by :func:`get_report_masks`.
        atypic_threshold: Share of rows below which a pattern is reported as
            atypical rather than a legitimate secondary format.

    Returns:
        A markdown report, suitable for saving under ``notebooks/8-reports``.
    """
    markdown = ["# Cleaning Report\n", mask_report.to_markdown()]

    for col in mask_report.columns:
        col_counts = mask_report[col]
        total = col_counts.sum()
        prevalent_pattern = str(col_counts.idxmax())
        prevalent_count = col_counts.max()

        classification = classify_pattern(prevalent_pattern)
        if classification == "Numeric":
            classification = (
                "Continuous numeric" if "." in prevalent_pattern else "Discrete numeric"
            )
        elif classification == "Alphabetic":
            classification = "Alphabetic categorical"
        elif classification == "Alphanumeric":
            classification = "Alphanumeric categorical"

        atypical = col_counts[
            (col_counts / mask_report.sum().iloc[0] < atypic_threshold) & (col_counts > 0)
        ]
        atypical_lines = [
            f"  - `{pat}` ({cnt} occurrence{'s' if cnt > 1 else ''})"
            for pat, cnt in atypical.items()
            if pat != "NaN"
        ]

        na_count = col_counts.get("NaN", 0)
        na_pct = (na_count / total * 100) if total > 0 else 0

        section = f"""
---
## Column: {col}
- **Prevalent pattern:** `{prevalent_pattern}` ({prevalent_count} occurrences).
- **Classification:** {classification}.
- **Atypical values:**
{chr(10).join(atypical_lines) if atypical_lines else "  - None"}
- **NA values:** {na_count} (≈{na_pct:.2f}%).
- **Conclusion:** Predominantly {classification.lower()}, with \
{"rare anomalies" if atypical_lines else "no anomalies"} and a \
{"small" if na_pct < 1 else "moderate"} proportion of missing values.
"""
        markdown.append(section)

    return "\n".join(markdown)


def clean_column_values(df: pd.DataFrame, regex: dict[str, str] | None = None) -> pd.DataFrame:
    """Replace values that don't match their column's expected regex with ``NA``.

    Args:
        df: The input DataFrame to be cleaned. Not mutated; a cleaned copy is
            returned.
        regex: Mapping of column name to a regex a *valid* value must fully
            match. Defaults to :data:`src.data.constants.RAW_COLUMN_REGEX`.

    Returns:
        A cleaned copy of ``df`` with atypical values replaced by ``NA``.
    """
    regex = RAW_COLUMN_REGEX if regex is None else regex
    df = df.copy()
    for column in df.columns:
        if column in regex:
            pattern = regex[column]
            df.loc[~df[column].astype(str).str.match(pattern), column] = pd.NA
    return df


def restrict_to_known_categories(
    df: pd.DataFrame, categories: dict[str, list[str]] | None = None
) -> pd.DataFrame:
    """Replace categorical values outside their known grade set with ``NA``.

    :func:`clean_column_values`'s regexes only check *shape* (e.g. ``color``
    accepts any single letter A-J, ``cut``/``clarity`` accept arbitrary
    alphabetic/alphanumeric text) — a syntactically valid but unknown grade
    (e.g. ``color="A"``, a ``cut`` typo) would otherwise survive cleaning and
    only fail later, inside ``OrdinalEncoder``, during model training or
    inference. This catches those values earlier, at the same cleaning step.

    Args:
        df: The input DataFrame to be cleaned. Not mutated; a cleaned copy is
            returned.
        categories: Mapping of column name to its list of known-valid
            values. Defaults to :data:`src.data.constants.ORDINAL_CATEGORIES`.

    Returns:
        A cleaned copy of ``df`` with out-of-vocabulary categorical values
        replaced by ``NA``.
    """
    categories = ORDINAL_CATEGORIES if categories is None else categories
    df = df.copy()
    for column, known_values in categories.items():
        if column in df.columns:
            df.loc[~df[column].isin(known_values), column] = pd.NA
    return df


def cast_clean_dtypes(df: pd.DataFrame, dtypes: dict[str, str] | None = None) -> pd.DataFrame:
    """Cast a fully-cleaned (no missing values) DataFrame to its final dtypes.

    Args:
        df: A DataFrame already free of missing values (see
            :func:`clean_column_values` followed by ``dropna``).
        dtypes: Mapping of column name to target dtype. Defaults to
            :data:`src.data.constants.CLEAN_DTYPES`.

    Returns:
        ``df`` cast to ``dtypes``.
    """
    dtypes = CLEAN_DTYPES if dtypes is None else dtypes
    return df.astype(dtypes)


def clean_diamantes(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full raw -> typed cleaning pipeline on a raw diamonds DataFrame.

    Replaces atypical values with ``NA`` (:func:`clean_column_values`),
    replaces categorical values outside their known grade set with ``NA``
    (:func:`restrict_to_known_categories` — a syntactically valid but unknown
    ``cut``/``color``/``clarity`` would otherwise slip through
    ``clean_column_values``'s shape-only regexes), drops the resulting
    incomplete rows, and casts the remaining columns to their final dtypes
    (:func:`cast_clean_dtypes`).

    Args:
        df: Raw diamonds DataFrame, as read from ``data/01_raw/diamantes.csv``.

    Returns:
        A cleaned, fully-typed DataFrame with no missing values.
    """
    cleaned = clean_column_values(df)
    cleaned = restrict_to_known_categories(cleaned)
    cleaned = cleaned.dropna().reset_index(drop=True)
    return cast_clean_dtypes(cleaned)
