"""Shared constants for the diamonds price-prediction project.

Single source of truth for column names, dtypes, ordinal category orders and
raw-data cleaning rules. Every pipeline stage (data cleaning, feature
engineering, model training, inference/serving) imports from here instead of
redefining these values locally, so the ordinal encoding, feature order and
dtype casting always stay consistent end to end.

See ``data/01_raw/datos_diamantes_Info.txt`` for the full data dictionary.
"""

# --- Column groups ------------------------------------------------------
#: Continuous numeric features used as-is.
NUMERIC_FEATURES: list[str] = ["carat", "depth", "table"]

#: Physical dimensions collapsed into a single ``volume`` feature (see
#: :func:`src.data.features.compute_volume`); kept separate from
#: ``NUMERIC_FEATURES`` because they go through an extra transformation step.
VOLUME_SOURCE_FEATURES: list[str] = ["x", "y", "z"]

#: Categorical features with a natural worst -> best order (see
#: ``ORDINAL_CATEGORIES`` below), encoded with ``OrdinalEncoder`` rather than
#: one-hot so the encoding preserves that order.
CATEGORICAL_ORDINAL_FEATURES: list[str] = ["cut", "color", "clarity"]

#: All model input features, in a fixed order.
FEATURE_COLUMNS: list[str] = (
    NUMERIC_FEATURES + VOLUME_SOURCE_FEATURES + CATEGORICAL_ORDINAL_FEATURES
)

#: Regression target.
TARGET_COLUMN: str = "price"

# --- Ordinal category orders (worst -> best) -----------------------------
# A higher position in each list means a higher expected price, matching the
# monotonic price effect found in the EDA (notebooks/2-exploration,
# notebooks/3-analysis). Used both for OrdinalEncoder (training) and for
# select widgets / validation in serving.
ORDINAL_CATEGORIES: dict[str, list[str]] = {
    "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    "color": ["J", "I", "H", "G", "F", "E", "D"],
    "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
}

# --- Raw-data cleaning rules (notebooks/3-analysis/02-jrz-...) -----------
#: Per-column regex a *valid* raw value must fully match; anything else is
#: treated as an atypical/corrupted value and replaced with ``NA``.
RAW_COLUMN_REGEX: dict[str, str] = {
    "carat": r"^[1-9](\.[0-9]+)?$",
    "cut": r"^[A-Za-z]+\s*[A-Za-z]*$",
    "color": r"^[A-Ja-j]$",
    "clarity": r"^[A-Za-z]{1,3}[0-9]{0,1}$",
    "depth": r"^[0-9]+\.[0-9]+$",
    "table": r"^[0-9]+\.[0-9]+$",
    "price": r"^[1-9][0-9]*$",
    "x": r"^(?=[0-9]{0,3}\.[0-9]+$)(?!0*\.0*$)[0-9]{0,3}\.[0-9]+$",
    "y": r"^(?=[0-9]{0,3}\.[0-9]+$)(?!0*\.0*$)[0-9]{0,3}\.[0-9]+$",
    "z": r"^(?=[0-9]{0,3}\.[0-9]+$)(?!0*\.0*$)[0-9]{0,3}\.[0-9]+$",
}

#: Expected number of ``NA`` values per column after
#: :func:`src.data.cleaning.clean_column_values`, used as a regression check
#: that the raw ``data/01_raw/diamantes.csv`` file hasn't silently changed.
EXPECTED_NA_COUNTS: dict[str, int] = {
    "carat": 70,
    "cut": 153,
    "color": 227,
    "clarity": 277,
    "depth": 307,
    "table": 251,
    "price": 273,
    "x": 142,
    "y": 147,
    "z": 54,
}

#: dtypes applied after cleaning and dropping ``NA`` rows.
CLEAN_DTYPES: dict[str, str] = {
    "carat": "float32",
    "cut": "category",
    "color": "category",
    "clarity": "category",
    "depth": "float32",
    "table": "float32",
    "price": "int64",
    "x": "float32",
    "y": "float32",
    "z": "float32",
}
