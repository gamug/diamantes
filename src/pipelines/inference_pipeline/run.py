"""Inference pipeline: new feature data + a trained model -> predictions.

Batch counterpart of ``notebooks/7-deploy/diamantes-streamlit-batch.py``'s
"Batch Prediction" tab, orchestrating :mod:`inference.predict` so it can run
without Streamlit (e.g. from a script or a scheduler).

Run with (``src`` must be importable, matching this project's pytest
``pythonpath`` convention — see ``CLAUDE.md``)::

    PYTHONPATH=src uv run python -m pipelines.inference_pipeline.run
"""

from pathlib import Path

import pandas as pd

from data.constants import FEATURE_COLUMNS
from inference.predict import load_production_model, predict_price, preprocess_batch_data

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

MODEL_DIR = DATA_DIR / "06_models"
INPUT_CSV_PATH = DATA_DIR / "05_model_input" / "diamantes_new.csv"
OUTPUT_CSV_PATH = DATA_DIR / "07_model_output" / "diamantes_predictions.csv"


def run(
    input_csv_path: Path = INPUT_CSV_PATH,
    output_csv_path: Path = OUTPUT_CSV_PATH,
    model_dir: Path = MODEL_DIR,
) -> pd.DataFrame:
    """Run batch inference on a CSV of new diamonds and write the predictions.

    Args:
        input_csv_path: CSV with (at least) the columns in
            :data:`data.constants.FEATURE_COLUMNS`, case/format tolerant
            (see :func:`inference.predict.preprocess_batch_data`).
        output_csv_path: Where to write the input rows plus a
            ``Predicted_Price`` column (``data/07_model_output``).
        model_dir: Directory containing the trained model
            (``data/06_models``).

    Returns:
        The input DataFrame with a ``Predicted_Price`` column appended.
    """
    model = load_production_model(model_dir)

    raw_df = pd.read_csv(input_csv_path)
    processed_df = preprocess_batch_data(raw_df)

    missing_cols = [col for col in FEATURE_COLUMNS if col not in processed_df.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing required columns: {missing_cols}")

    predictions = predict_price(model, processed_df)

    result_df = raw_df.copy()
    result_df["Predicted_Price"] = predictions

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv_path, index=False)
    print(f"Wrote {len(result_df)} predictions to {output_csv_path}")

    return result_df


if __name__ == "__main__":
    run()
