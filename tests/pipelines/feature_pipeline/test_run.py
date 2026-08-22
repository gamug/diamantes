from pathlib import Path

from data.extraction import load_parquet
from pipelines.feature_pipeline.run import (
    FEATURE_PARQUET_PATH,
    INTERMEDIATE_PARQUET_PATH,
    RAW_CSV_PATH,
    run,
)


def test_feature_pipeline_run_produces_clean_feature_data(tmp_path: Path) -> None:
    raw_csv_path = tmp_path / "diamantes.csv"
    raw_csv_path.write_text(
        "carat,cut,color,clarity,depth,table,price,x,y,z\n"
        "1.01,Ideal,G,VS2,61.8,57.0,4268,6.42,6.45,3.98\n"  # valid
        "1.01,Ideal,G,VS2,61.8,57.0,4268,6.42,6.45,3.98\n"  # duplicate of row above
        "bad,Premium,H,SI1,62.0,58.0,1500,7.32,7.28,4.53\n"  # atypical carat -> dropped by cleaning
        "2.00,Fair,J,SI2,63.0,54.0,3000,6.00,6.10,2.00\n"  # z == carat -> corrupted, dropped
    )

    intermediate_path = tmp_path / "02_intermediate" / "diamantes_type_fixed.parquet"
    feature_path = tmp_path / "04_feature" / "diamantes_clean.parquet"

    run(
        raw_csv_path=raw_csv_path,
        intermediate_parquet_path=intermediate_path,
        feature_parquet_path=feature_path,
    )

    assert intermediate_path.exists()
    assert feature_path.exists()

    feature_df = load_parquet(feature_path)
    assert len(feature_df) == 1
    assert feature_df.iloc[0]["cut"] == "Ideal"


def test_feature_pipeline_run_uses_default_paths_signature() -> None:
    # sanity check the default paths are wired to the expected data/ layers
    assert RAW_CSV_PATH.parts[-2:] == ("01_raw", "diamantes.csv")
    assert INTERMEDIATE_PARQUET_PATH.parts[-2:] == (
        "02_intermediate",
        "diamantes_type_fixed.parquet",
    )
    assert FEATURE_PARQUET_PATH.parts[-2:] == ("04_feature", "diamantes_clean.parquet")
