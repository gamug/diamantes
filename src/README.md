# Feature/Training/Inference Pipelines

File Structure based on:

<https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines>

## Folder Structure

- src: source code
    - data: data extraction, data validation, data processing, data transformation, data save and export, etc.
    - model: model training, model evaluation, model validation, model save and export, etc.
    - inference: model prediction, model serving, model monitoring, etc.
    - pipelines:
        - feature_pipeline: takes as input raw data that it transforms into features (and labels)
        - training_pipeline: takes as input features and labels that it transforms into a model
        - inference_pipeline: takes new feature data and a trained model and makes predictions.

you could have multiple pipelines, for example:

- 3 feature pipelines that extract raw data from different sources and transform them into features and save it into a feature store.
- 2 training pipelines that take the features from the feature store and train different models.
- 3 inference pipeline that creates a model serving endpoint for each of the trained models and 1 batch
  inference pipeline that takes the features from the feature store and makes predictions in batch mode.

Finally is recommended to have a script that orchestrates the execution of the pipelines. This script should could be run in a cron job or a workflow orchestrator like Airflow, Prefect, Dagster, etc.

---

## This project's modules

Concrete implementation for the diamonds price-prediction task. Every module is unit-tested under the
mirrored path in `tests/` (e.g. `src/data/cleaning.py` ↔ `tests/data/test_cleaning.py`).

| Module | What it does |
|---|---|
| `data/constants.py` | Single source of truth: feature column groups, `ORDINAL_CATEGORIES` (worst→best grade orders), raw-data cleaning regexes/dtypes. Every other module imports from here instead of redefining these values. |
| `data/extraction.py` | `load_raw_diamantes`, `save_parquet`/`load_parquet` — read the raw CSV, read/write the `data/02_intermediate` and `data/04_feature` Parquet layers. |
| `data/cleaning.py` | Raw → typed: `clean_diamantes` (mask atypical values with regex, drop unknown categorical grades, cast dtypes) plus the lower-level pieces it's built from and the mask/report helpers (`get_all_masks`, `get_report_cleaning`, ...). |
| `data/features.py` | Typed → features: `remove_duplicate_rows`, `remove_corrupted_rows` (column-swap data-entry errors), `build_preprocessor` (the `ColumnTransformer`: impute + scale numerics, collapse `x,y,z` into `volume`, ordinal-encode `cut`/`color`/`clarity`). |
| `data/validation.py` | **Not yet implemented** — reserved for the `deepchecks` data-integrity / train-test-validation suites. See the module docstring for the planned shape. |
| `model/baseline.py` | `HeuristicPriceRegressor` — the rule-based (non-ML) pricing baseline every trained model should clear by a wide margin. |
| `model/train.py` | `build_model_pipeline`, `split_train_test`, `train_model` — the `TransformedTargetRegressor(log-price) + HistGradientBoostingRegressor` pipeline and its `GridSearchCV` hyperparameter search. |
| `model/evaluate.py` | `regression_metrics` — MAE/RMSE/MAPE/R² in one call, MAPE being this project's primary metric (see `notebooks/1-data`). |
| `model/persistence.py` | `save_model`/`load_model` — joblib save/load for a fitted pipeline. |
| `model/validation.py` | **Not yet implemented** — reserved for the `deepchecks` model-evaluation suite. See the module docstring for the planned shape. |
| `inference/predict.py` | `load_production_model`, `predict_price`, `preprocess_batch_data`, `is_out_of_training_range` — the business logic behind both Streamlit apps in `notebooks/7-deploy`. |
| `inference/explain.py` | `explain_prediction`, `baseline_price` — the SHAP-based, multiplicative "why this price?" breakdown used by the single-diamond Streamlit app. |
| `pipelines/feature_pipeline/run.py` | Orchestrates `data.*`: raw CSV → cleaned/typed (`data/02_intermediate`) → deduplicated/feature-ready (`data/04_feature`). |
| `pipelines/training_pipeline/run.py` | Orchestrates `model.*`: engineered features → split → tuned model → test-set metrics → saved artifact (`data/06_models`). |
| `pipelines/inference_pipeline/run.py` | Orchestrates `inference.*`: a CSV of new diamonds + the saved model → a predictions CSV (`data/07_model_output`). |

There is no single `app.py` entry point — the two Streamlit apps under `notebooks/7-deploy/` *are* the
inference-serving entry points (see "Consuming this library from outside `src/`" below), and each of the
three pipelines above is its own runnable script (see next section).

## Import convention

`pyproject.toml` sets `pythonpath = ["src"]` for pytest, so modules are imported by their **bare top-level
name** — `data.constants`, `model.train`, `inference.predict`, `pipelines.feature_pipeline.run` — never with a
`src.` prefix (there's no `src/__init__.py`; `src` itself isn't a package, it's a root added to the import
path). `.code_quality/mypy.ini` sets the matching `mypy_path = src` so `mypy` resolves the same imports.

```python
# correct
from data.constants import FEATURE_COLUMNS
from model.train import train_model

# wrong — do not do this
from src.data.constants import FEATURE_COLUMNS
```

## Running the pipelines

Each pipeline is a plain script with a `run()` function and a `if __name__ == "__main__": run()` guard, so it
can be run as a module (needs `src` on the import path — either export `PYTHONPATH=src` once per shell, or
prefix each command as below) or imported and called from Python/a notebook.

```bash
# 1. raw CSV -> data/02_intermediate -> data/04_feature
PYTHONPATH=src uv run python -m pipelines.feature_pipeline.run

# 2. data/04_feature -> tuned model -> data/06_models (prints the held-out test metrics)
PYTHONPATH=src uv run python -m pipelines.training_pipeline.run

# 3. a CSV of new diamonds + data/06_models -> data/07_model_output/diamantes_predictions.csv
#    (put your input CSV at data/05_model_input/diamantes_new.csv first, or pass paths explicitly — see below)
PYTHONPATH=src uv run python -m pipelines.inference_pipeline.run
```

Equivalent `make` targets:

```bash
make build_features   # -> pipelines.feature_pipeline.run
make train_model       # -> pipelines.training_pipeline.run
make predict_batch     # -> pipelines.inference_pipeline.run
```

Every `run()` accepts keyword overrides for its input/output paths (and, for training, the hyperparameter grid
and CV folds), which is how the test suite exercises them against temporary files instead of the real `data/`
layers:

```python
from pathlib import Path
from pipelines.training_pipeline.run import run

metrics = run(
    feature_parquet_path=Path("data/04_feature/diamantes_clean.parquet"),
    model_dir=Path("data/06_models"),
    model_file_name="diamantes_price-hist_gradient_boosting-v1.joblib",
    hyperparameters={"regressor__model__max_iter": [200]},  # smaller grid, e.g. for a quick local run
    cv=3,
)
print(metrics)  # {'MAE': ..., 'RMSE': ..., 'MAPE_%': ..., 'R2': ...}
```

## Usage examples

Train (or retrain) and evaluate the model without going through the pipeline wrapper:

```python
from data.extraction import load_parquet
from data.constants import FEATURE_COLUMNS, TARGET_COLUMN
from model.train import split_train_test, train_model
from model.evaluate import regression_metrics
from model.persistence import save_model
from pathlib import Path

dataset = load_parquet(Path("data/04_feature/diamantes_clean.parquet"))
x_train, x_test, y_train, y_test = split_train_test(dataset[FEATURE_COLUMNS], dataset[TARGET_COLUMN])

grid_search = train_model(x_train, y_train)  # full GridSearchCV, see DEFAULT_HYPERPARAMETER_GRID
best_model = grid_search.best_estimator_

print(regression_metrics(y_test, best_model.predict(x_test)))
save_model(best_model, Path("data/06_models/my-model.joblib"))
```

Score a new diamond and explain the prediction (this is exactly what
`notebooks/7-deploy/diamantes-streamlit.py` does):

```python
import pandas as pd
from pathlib import Path
from inference.predict import load_production_model, predict_price
from inference.explain import explain_prediction

model = load_production_model(Path("data/06_models"))
diamond = pd.DataFrame([{
    "carat": 1.0, "depth": 61.5, "table": 57.0,
    "x": 6.4, "y": 6.5, "z": 4.0,
    "cut": "Ideal", "color": "G", "clarity": "SI1",
}])

price = predict_price(model, diamond)[0]
breakdown = explain_prediction(model, diamond)  # per-feature price multiplier, sorted by impact
```

Clean a raw batch of diamonds by hand (what `pipelines.feature_pipeline.run` does internally):

```python
from pathlib import Path
from data.extraction import load_raw_diamantes
from data.cleaning import clean_diamantes
from data.features import remove_duplicate_rows, remove_corrupted_rows

raw_df = load_raw_diamantes(Path("data/01_raw/diamantes.csv"))
typed_df = clean_diamantes(raw_df)                       # regex-clean atypical/unknown values, drop, cast dtypes
feature_df = remove_corrupted_rows(remove_duplicate_rows(typed_df))
```

## Consuming this library from outside `src/`

`notebooks/7-deploy/diamantes-streamlit.py` and `diamantes-streamlit-batch.py` are run directly by Streamlit
(`streamlit run <path>`), which doesn't know about pytest's `pythonpath` setting — so each script inserts `src`
onto `sys.path` itself, once, before importing anything from this library:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.constants import FEATURE_COLUMNS, ORDINAL_CATEGORIES
from inference.predict import load_production_model, predict_price
```

Run either app with:

```bash
uv run streamlit run notebooks/7-deploy/diamantes-streamlit.py         # single-diamond predictor
uv run streamlit run notebooks/7-deploy/diamantes-streamlit-batch.py   # + batch CSV upload tab
```

Any other standalone script (a notebook, a new CLI, a future API) that wants to import from `src/` should use
the same `sys.path.insert(...)` pattern, or run under `PYTHONPATH=src` as shown above.

## Testing

```bash
make test                                          # everything, with coverage
uv run pytest tests/data/test_cleaning.py -v       # a single module
uv run pytest tests/model -v                       # a whole layer
```

Tests import the same way the library does — bare top-level names (`from data.cleaning import clean_diamantes`)
— because they run under the same `pythonpath = ["src"]` pytest setting.
