from pathlib import Path

import pandas as pd

from data.extraction import load_parquet, load_raw_diamantes, save_parquet


def test_load_raw_diamantes(tmp_path: Path) -> None:
    csv_path = tmp_path / "diamantes.csv"
    csv_path.write_text("carat,cut,price\n1.01,Ideal,4268\n0.5,Premium,1500\n")

    df = load_raw_diamantes(csv_path)

    assert list(df.columns) == ["carat", "cut", "price"]
    assert len(df) == 2


def test_save_and_load_parquet_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"carat": [1.01, 0.5], "cut": pd.Categorical(["Ideal", "Premium"])})
    parquet_path = tmp_path / "nested" / "diamantes.parquet"

    save_parquet(df, parquet_path)
    assert parquet_path.exists()

    loaded = load_parquet(parquet_path)
    pd.testing.assert_frame_equal(loaded, df)
