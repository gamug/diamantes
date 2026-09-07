"""Feature pipeline: raw diamonds CSV -> model-ready feature table.

Standalone, autonomously runnable script for issue #22 ("1. Feature pipeline
development"), part of the FTI scripts architecture (issue #15).

It:

1. **reads** the raw data from the source defined for the project
   (``data/01_raw/diamantes.csv`` -- the modified file supplied for the
   course, never the original Kaggle dataset);
2. **transforms** it into features suitable for the price model, using the
   pure functions in :mod:`diamond_features` (type-fix the text numeric
   columns, validate against the data dictionary, drop invalid / duplicate /
   column-swap rows, engineer ``volume = x * y * z``);
3. **stores** the processed features as a Parquet file in the feature layer
   (``data/04_feature/diamantes_features.parquet``).

Run it (``src`` is on ``sys.path`` via ``pyproject.toml``'s
``[tool.pytest.ini_options] pythonpath``; for a bare ``python`` call, either
``cd src`` or set ``PYTHONPATH=src``)::

    uv run python src/feature_pipeline.py
    uv run python -m feature_pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from diamond_features import build_features

logger = logging.getLogger(__name__)

#: Repository root: this file lives at ``src/feature_pipeline.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_CSV_PATH = DATA_DIR / "01_raw" / "diamantes.csv"
FEATURE_PARQUET_PATH = DATA_DIR / "04_feature" / "diamantes_features.parquet"


def load_raw_data(csv_path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the raw diamonds CSV with no type coercion.

    Args:
        csv_path: Path to ``data/01_raw/diamantes.csv``.

    Returns:
        The raw table, every column read as text so the transformation step
        can inspect the original string values.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {csv_path}")
    return pd.read_csv(csv_path, dtype="object", low_memory=False)


def save_features(df: pd.DataFrame, path: Path = FEATURE_PARQUET_PATH) -> None:
    """Write the feature table to Parquet, creating parent directories.

    Args:
        df: Feature table from :func:`diamond_features.build_features`.
        path: Destination ``.parquet`` file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run_feature_pipeline(
    raw_csv_path: Path = RAW_CSV_PATH,
    output_path: Path = FEATURE_PARQUET_PATH,
) -> pd.DataFrame:
    """Load the raw CSV, build the feature table and persist it.

    Args:
        raw_csv_path: Source raw CSV.
        output_path: Destination Parquet file for the feature table.

    Returns:
        The feature table that was written (also handy for tests / notebooks).
    """
    logger.info("Loading raw data from %s", raw_csv_path)
    raw_df = load_raw_data(raw_csv_path)
    logger.info("Raw data: %d rows x %d columns", raw_df.shape[0], raw_df.shape[1])

    features = build_features(raw_df)
    dropped = raw_df.shape[0] - features.shape[0]
    logger.info(
        "Feature table: %d rows x %d columns (%d raw rows dropped as invalid/duplicate)",
        features.shape[0],
        features.shape[1],
        dropped,
    )

    save_features(features, output_path)
    logger.info("Wrote feature table to %s", output_path)
    return features


def main() -> None:
    """Console entry point: configure logging and run the pipeline once."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    run_feature_pipeline()


if __name__ == "__main__":
    main()
