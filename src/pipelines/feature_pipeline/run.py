"""Feature pipeline: raw diamonds data -> cleaned, typed, feature-engineered data.

Orchestrates :mod:`data.extraction`, :mod:`data.cleaning` and
:mod:`data.features` to reproduce, as a single runnable script, the chain of
notebooks ``notebooks/3-analysis/02-jrz-data_description_Manual-pandas-...``
(raw -> ``02_intermediate``) and
``notebooks/4-feat_eng/01-gmg-basic-feature-engineering-pipeline-...``
(``02_intermediate`` -> ``04_feature``).

Run with (``src`` must be importable, matching this project's pytest
``pythonpath`` convention — see ``CLAUDE.md``)::

    PYTHONPATH=src uv run python -m pipelines.feature_pipeline.run

TODO(tomorrow): once :mod:`data.validation` is implemented, run
``check_data_integrity``/``check_train_test_split`` here and fail the
pipeline (or at least warn loudly) before writing the feature layer.
"""

from pathlib import Path

from data.cleaning import clean_diamantes
from data.extraction import load_parquet, load_raw_diamantes, save_parquet
from data.features import remove_corrupted_rows, remove_duplicate_rows

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

RAW_CSV_PATH = DATA_DIR / "01_raw" / "diamantes.csv"
INTERMEDIATE_PARQUET_PATH = DATA_DIR / "02_intermediate" / "diamantes_type_fixed.parquet"
FEATURE_PARQUET_PATH = DATA_DIR / "04_feature" / "diamantes_clean.parquet"


def run(
    raw_csv_path: Path = RAW_CSV_PATH,
    intermediate_parquet_path: Path = INTERMEDIATE_PARQUET_PATH,
    feature_parquet_path: Path = FEATURE_PARQUET_PATH,
) -> None:
    """Run the full feature pipeline and write both the intermediate and feature layers.

    Args:
        raw_csv_path: Source raw CSV (``data/01_raw/diamantes.csv``).
        intermediate_parquet_path: Where to write the cleaned/typed data
            (``data/02_intermediate``).
        feature_parquet_path: Where to write the deduplicated,
            corrupted-row-free feature data (``data/04_feature``).
    """
    raw_df = load_raw_diamantes(raw_csv_path)
    print(f"Loaded raw data: {raw_df.shape}")

    typed_df = clean_diamantes(raw_df)
    print(f"Cleaned + typed data: {typed_df.shape}")
    save_parquet(typed_df, intermediate_parquet_path)

    typed_df = load_parquet(intermediate_parquet_path)
    deduped_df = remove_duplicate_rows(typed_df)
    feature_df = remove_corrupted_rows(deduped_df)
    print(
        f"Feature data: {feature_df.shape} "
        f"(dropped {len(typed_df) - len(feature_df)} duplicate/corrupted rows)"
    )
    save_parquet(feature_df, feature_parquet_path)


if __name__ == "__main__":
    run()
