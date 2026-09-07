"""Data validation and integrity checks for the diamond feature table (issue #23).

The feature pipeline must guarantee data quality *before* it persists anything:
:func:`validate_features` is the contract the built feature table has to satisfy,
and :func:`feature_pipeline.run_feature_pipeline` calls it between
``build_features`` and ``save_features`` so a failing table is never written to
``data/04_feature/``.

Rules enforced (implemented with `pandera <https://pandera.readthedocs.io/>`_):

- **Types** -- every column has its expected dtype (``float32`` measurements,
  ``category`` grades, ``int64`` price); no silent coercion.
- **Ranges** -- each physical measurement stays within the span documented in
  ``data/01_raw/datos_diamantes_Info.txt``; ``price`` and ``volume`` are
  strictly positive.
- **Valid categories** -- ``cut`` / ``color`` / ``clarity`` only take a value
  from their known grade vocabulary.
- **Nulls** -- at most :data:`MAX_NULL_FRACTION` of a column may be missing
  (0% by default: the pipeline is expected to produce a complete table).
- **Uniqueness** -- no two rows are identical (a catalog row is its own key;
  duplicates would leak across a later train/test split).
- **Field integrity** -- ``volume == x * y * z``; ``depth`` agrees with the
  geometry ``2*z / (x + y) * 100`` that defines it (within
  :data:`DEPTH_CONSISTENCY_TOLERANCE_PP`); no ``z == carat`` / ``y == depth``
  column-swap rows survive.

The reference course material also lists *date formats* as a validation
dimension; the diamonds dataset has no temporal fields, so that rule is not
applicable here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from diamond_features import (
    FEATURE_TABLE_COLUMNS,
    KNOWN_CATEGORIES,
    MEASUREMENT_BOUNDS,
    PRICE_COLUMN,
    VOLUME_COLUMN,
)

#: Maximum fraction of missing values tolerated in any single column.
MAX_NULL_FRACTION: float = 0.0

#: Allowed absolute disagreement (percentage points) between the recorded
#: ``depth`` and the depth percentage implied by the geometry
#: ``2*z / (x + y) * 100``. Slightly looser than the cleaning threshold in
#: :func:`diamond_features.drop_geometry_inconsistent_rows` so validation is a
#: safety net, not a duplicate of the cleaning step.
DEPTH_CONSISTENCY_TOLERANCE_PP: float = 2.0

#: Relative tolerance for the ``volume == x * y * z`` check (float32 round-trip).
_VOLUME_RTOL: float = 1e-3


class FeatureValidationError(RuntimeError):
    """Raised when the feature table fails a data-quality or integrity rule.

    The message lists every failing rule so the pipeline operator can see, in
    one place, why the features were rejected and not persisted.
    """


def _volume_matches_dimensions(df: pd.DataFrame) -> pd.Series:
    """``volume`` equals ``x * y * z`` (within float tolerance) for every row."""
    expected = df["x"] * df["y"] * df["z"]
    close = np.isclose(df[VOLUME_COLUMN], expected, rtol=_VOLUME_RTOL, atol=_VOLUME_RTOL)
    return pd.Series(close, index=df.index)


def _depth_matches_geometry(df: pd.DataFrame) -> pd.Series:
    """Recorded ``depth`` agrees with ``2*z / (x + y) * 100`` for every row."""
    implied = 2.0 * df["z"] / (df["x"] + df["y"]) * 100.0
    return (df["depth"] - implied).abs() <= DEPTH_CONSISTENCY_TOLERANCE_PP


def build_feature_schema(max_null_fraction: float = MAX_NULL_FRACTION) -> pa.DataFrameSchema:
    """Build the pandera schema for a model-ready diamond feature table.

    Args:
        max_null_fraction: Passed through so the column ``nullable`` flag stays
            consistent with the null-fraction gate in :func:`validate_features`
            (columns are non-nullable only when no missing values are allowed).

    Returns:
        A :class:`pandera.pandas.DataFrameSchema` covering types, ranges, valid
        categories, row uniqueness and cross-field integrity.
    """
    nullable = max_null_fraction > 0.0

    # Columns are declared in FEATURE_TABLE_COLUMNS order so ``ordered=True``
    # also validates the layout of the table.
    columns: dict[str, pa.Column] = {}
    for name in FEATURE_TABLE_COLUMNS:
        if name in MEASUREMENT_BOUNDS:
            low, high = MEASUREMENT_BOUNDS[name]
            columns[name] = pa.Column("float32", pa.Check.in_range(low, high), nullable=nullable)
        elif name in KNOWN_CATEGORIES:
            columns[name] = pa.Column(
                "category", pa.Check.isin(KNOWN_CATEGORIES[name]), nullable=nullable
            )
        elif name == VOLUME_COLUMN:
            columns[name] = pa.Column("float32", pa.Check.gt(0.0), nullable=nullable)
        elif name == PRICE_COLUMN:
            columns[name] = pa.Column("int64", pa.Check.gt(0), nullable=nullable)
        else:  # pragma: no cover - guard against an unhandled feature column
            raise ValueError(f"no validation rule defined for feature column {name!r}")

    return pa.DataFrameSchema(
        columns,
        strict=True,
        ordered=True,
        coerce=False,
        unique=list(FEATURE_TABLE_COLUMNS),
        checks=[
            pa.Check(_volume_matches_dimensions, name="volume_equals_x_times_y_times_z"),
            pa.Check(_depth_matches_geometry, name="depth_matches_geometry"),
            pa.Check(lambda df: df["z"] != df["carat"], name="no_z_carat_column_swap"),
            pa.Check(lambda df: df["y"] != df["depth"], name="no_y_depth_column_swap"),
        ],
    )


def _check_null_fraction(df: pd.DataFrame, max_null_fraction: float) -> None:
    """Raise :class:`FeatureValidationError` if any column is too sparse."""
    null_fraction = df.isna().mean()
    offenders = null_fraction[null_fraction > max_null_fraction]
    if not offenders.empty:
        detail = ", ".join(f"{col}={frac:.1%}" for col, frac in offenders.items())
        raise FeatureValidationError(
            f"null fraction exceeds the allowed {max_null_fraction:.1%}: {detail}"
        )


def validate_features(
    df: pd.DataFrame, max_null_fraction: float = MAX_NULL_FRACTION
) -> pd.DataFrame:
    """Validate a feature table against the full data-quality / integrity contract.

    Args:
        df: The feature table produced by
            :func:`diamond_features.build_features`.
        max_null_fraction: Maximum share of missing values allowed per column.

    Returns:
        The same DataFrame, unchanged, when every rule passes (so callers can
        write ``features = validate_features(features)``).

    Raises:
        FeatureValidationError: If any rule fails. The message enumerates every
            failing rule; the feature table must not be persisted.
    """
    _check_null_fraction(df, max_null_fraction)

    schema = build_feature_schema(max_null_fraction)
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases
        summary = (
            cases.groupby("check")["failure_case"].agg(["count", "first"])
            if not cases.empty
            else cases
        )
        raise FeatureValidationError(
            f"feature table failed {len(cases)} validation check(s):\n{summary}"
        ) from exc
