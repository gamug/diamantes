"""Data extraction and layered save/load helpers.

Reads the raw diamonds CSV and reads/writes the intermediate and feature
Parquet layers described in ``data/README.md``
(``01_raw`` -> ``02_intermediate`` -> ... -> ``04_feature``).
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa


def load_raw_diamantes(csv_path: Path) -> pd.DataFrame:
    """Load the raw diamonds CSV, exactly as supplied for the course.

    Args:
        csv_path: Path to ``data/01_raw/diamantes.csv``. Do not point this at
            the original Kaggle dataset (see project ``CLAUDE.md``).

    Returns:
        The raw DataFrame, with no type coercion applied (all columns are
        read as-is so cleaning can inspect the original string values).
    """
    return pd.read_csv(csv_path, low_memory=False)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to Parquet, inferring the schema from ``df`` itself.

    Args:
        df: The DataFrame to persist.
        path: Destination ``.parquet`` file; parent directories are created
            if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.Table.from_pandas(df).schema
    df.to_parquet(path, index=False, schema=schema)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a DataFrame previously saved with :func:`save_parquet`.

    Args:
        path: Path to a ``.parquet`` file.

    Returns:
        The loaded DataFrame.
    """
    return pd.read_parquet(path, engine="pyarrow")
