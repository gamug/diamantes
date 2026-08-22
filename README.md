# 💎 Diamond Price Prediction

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

Academic project for the **Módulo de Ciencia de datos en producción**: predict the price of a diamond (`price`, in USD)
from its physical and quality characteristics (`carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`,
`z`). Built on [JoseRZapata's data-science-project-template](https://github.com/JoseRZapata/data-science-project-template)
(managed via [Cruft]), following the [FTI (Feature/Training/Inference) pipeline pattern](https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines)
and a [Kedro-style layered data-engineering convention][Data structure].

This README documents the project itself — the problem, the dataset, and the findings produced by every notebook
from `1-data` through `7-deploy`. For the generic template features (tooling rationale, devcontainer, CI,
etc.) see the [upstream template docs](https://joserzapata.github.io/data-science-project-template/#features-and-tools).

## 📦 The dataset

- **Source**: `data/01_raw/diamantes.csv` — a **modified** version of the public
  [Kaggle diamonds dataset](https://www.kaggle.com/shivam2503/diamonds), supplied for this course.
  **The original Kaggle dataset must not be used**; only this modified CSV is authoritative
  (see `data/01_raw/datos_diamantes_Info.txt`).
- **Target**: `price` — price in US dollars (\$326–\$18,823 in the cleaned data).
- **Features**: `carat` (weight), `cut`/`color`/`clarity` (ordinal quality grades), `depth`/`table` (proportions,
  %), `x`/`y`/`z` (physical dimensions, mm).
- **Snapshot**: static, single delivery for the course — no update mechanism, no streaming/online-learning
  concerns (see `notebooks/1-data`).
- **Framing** (see `notebooks/1-data`): supervised, **batch/offline regression**. The chosen headline metric is
  **MAPE** (Mean Absolute Percentage Error) rather than a raw dollar error, because prices span two orders of
  magnitude — a fixed dollar error means something very different on a \$400 stone than an \$18,000 one. The
  informal human/industry baseline is manual gemologist appraisal against a Rapaport-style price list.

## 🚀 Quick start

```bash
make init_git         # initialize git (if not already a repo)
make install_env       # uv sync --all-groups + install pre-commit hooks
source .venv/bin/activate
```

To add a new dependency:

```bash
uv add <package-name>              # runtime dependency
uv add --group dev <package-name>  # dev-only dependency
```

Run tests / checks:

```bash
make test    # pytest --cov
make check   # ruff + mypy + commitizen via pre-commit, all files
make lint    # ruff only
```

Run a notebook headlessly (used throughout this project to verify every notebook executes end-to-end with 0
errors before being committed):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 <notebook>.ipynb
```

## 🗃️ Project structure

```bash
.
├── data
│   ├── 01_raw                          # diamantes.csv (immutable source of truth)
│   ├── 02_intermediate                 # diamantes_type_fixed.parquet (typed, 19,231 rows)
│   ├── 03_primary                      # (unused — cleaning happens in 02_intermediate / 04_feature)
│   ├── 04_feature                      # diamantes_clean.parquet (deduped + corrupted rows removed, 18,977 rows)
│   ├── 05_model_input                  # (unused — preprocessing is embedded in each model's sklearn Pipeline)
│   ├── 06_models                       # 🏆 final serialized model artifacts (joblib + MLflow format)
│   ├── 07_model_output                 # (not yet used — see Roadmap)
│   ├── 08_reporting                    # ydata-profiling HTML report
│   └── README.md                       # description of the data layering convention
├── notebooks
│   ├── 1-data                          # problem framing (Géron-style questions)
│   ├── 2-exploration                   # data types, cleaning, first EDA
│   ├── 3-analysis                      # univariate / bivariate / multivariate EDA, estimator selection, heuristic model
│   ├── 4-feat_eng                      # deduplication, outlier policy, feature pipeline, stratified split
│   ├── 5-models                        # baseline → model selection → tuning → MLflow tracking → Deepchecks validation
│   ├── 6-interpretation                # permutation importance, SHAP, PDP/ICE, weak-segment analysis
│   ├── 7-deploy                        # 2 Streamlit apps: single + batch price prediction, SHAP explanation
│   └── 8-reports                       # ⏳ not started yet
├── models                              # unused (template default) — final models live in data/06_models instead
├── src                                 # FTI pipeline source code — data/model/inference + 3 runnable pipelines, see src/README.md
├── pyproject.toml                      # dependencies (uv-managed)
└── README.md                           # this file
```

## 🔬 Pipeline walkthrough & findings

Each stage below links to its notebook(s) and summarizes what was actually found — not just what the stage does.
Every number quoted here was produced by re-executing the corresponding notebook end-to-end.

### 1 · `notebooks/1-data` — Problem framing

[`01-gmg-Carga_de_datos-2026_08_15.ipynb`](notebooks/1-data/01-gmg-Carga_de_datos-2026_08_15.ipynb)

Answers Aurélien Géron's (*Hands-On ML*) problem-framing questionnaire for this dataset:

- **Objective**: a regression model estimating `price`, ultimately meant to reduce the information asymmetry
  between diamond sellers and buyers (an interactive **Streamlit app** is the envisioned delivery vehicle — see
  `7-deploy`).
- **Learning setup**: supervised, batch/offline (the data is a static one-time snapshot for the course).
- **Performance metric**: MAPE, chosen over RMSE/MAE because of the wide price range; a useful model should beat
  the ~5–10% negotiation margin typical of manual diamond appraisal.
- **Comparable problems**: hedonic/heterogeneous-good price prediction (e.g. house-price regression) — existing
  tooling and architecture patterns transfer directly.
- **Assumptions logged up front**: the modified CSV preserves the real physical/price relationships of the
  original data; the dataset is representative enough of the market to generalize.

### 2 · `notebooks/2-exploration` — Data typing & cleaning

[`01-gmg-data_explore_description-2026_08_18.ipynb`](notebooks/2-exploration/01-gmg-data_explore_description-2026_08_18.ipynb)

The raw CSV loads with **55,126 rows**, but every numeric column is read as `object` (string) — a common
symptom of stray non-numeric characters or malformed number formats mixed into the column. A regex-based mask
scan identifies and cleans the malformed values, converting the unparseable ones to `NaN`.

- After cleaning, missing values are heavily concentrated in `carat` (tens of thousands of rows) plus a smaller,
  roughly uniform spread of a few hundred `NaN`s across the other columns.
- **Decision**: drop every row with any remaining `NaN` (the dataset is large enough to absorb the loss without
  hurting downstream analysis) → **19,231 rows survive**, saved as
  `data/02_intermediate/diamantes_type_fixed.parquet`. This typed, deduplicated-of-nulls file is the starting
  point for every notebook downstream.
- Columns are typed: `carat`/`depth`/`table`/`x`/`y`/`z` → `float32`, `price` → `int64`, `cut`/`color`/`clarity`
  → pandas `category`. No boolean, nominal-categorical, or free-text string columns exist in this dataset.
- **Scope note carried through the whole project**: the cleaned data's `carat` values start at **1.00** (not
  0.2 as the original data dictionary's stated range implies) — this dataset over-represents larger stones, and
  predictions for `carat < 1.00` should be treated as extrapolation.
- No non-positive (`<= 0`) values remain in any numeric column after cleaning.

### 3 · `notebooks/3-analysis` — Exploratory data analysis

Three notebooks, increasing in depth:

- [`01-gmg-data_description_YData Profiling-2026_08_18.ipynb`](<notebooks/3-analysis/01-gmg-data_description_YData Profiling-2026_08_18.ipynb>)
  — automated profiling via `ydata-profiling` (report saved to `data/08_reporting/`); useful for a fast
  first pass, but bivariate/multivariate relationships still had to be analyzed manually.
- [`03-gmg-univariate-data_description_Manual-pandas-2026_08_18.ipynb`](notebooks/3-analysis/03-gmg-univariate-data_description_Manual-pandas-2026_08_18.ipynb)
  — manual univariate pass. Key findings: `price` is strongly right-skewed (median \$2,443 vs. mean \$3,928);
  `carat`/`x`/`y`/`z` share the same right-skew (small stones dominate); `depth`/`table` are the best-behaved,
  roughly symmetric numeric features; `cut` is imbalanced toward **Ideal** (~40%), `clarity` is imbalanced at
  both extremes (**IF** ~3%, **I1** ~1.4%), `color` is fairly balanced across its 7 grades.
- [`02-jrz-data_description_Manual-pandas-2024_10_24.ipynb`](notebooks/3-analysis/02-jrz-data_description_Manual-pandas-2024_10_24.ipynb)
  — the deepest notebook: bivariate + multivariate analysis, estimator selection, and a heuristic pricing model.

**Bivariate findings (target vs. features):**

| Relationship | Finding |
| --- | --- |
| `price` vs. `carat`/`x`/`z`/`y` | Strong, convex correlation (Pearson r ≈ 0.79 / 0.79 / 0.77 / 0.69) — the single biggest price driver, but non-linear (bigger jumps in price per extra carat at higher carat). |
| `price` vs. `depth`/`table` | Very weak (r ≈ −0.07 / −0.03) — proportions barely move price on their own. |
| `price` vs. `clarity`/`cut`/`color` | Ranked by stratification strength: **clarity > cut > color**. Raw `price` vs. `color`/`cut` looks *non-monotonic* at first glance — this is **Simpson's paradox**: cheaper, larger-carat stones are unevenly distributed across grades, and the paradox disappears once you control for carat (see `price_per_carat` below). |

**Multivariate findings:**

- `carat`, `x`, `y`, `z` are pairwise correlated at **r = 0.84–0.98** — severe multicollinearity, since they are
  all measuring the same physical size from different angles.
- `depth` and `table` anti-correlate (r ≈ −0.29) — a known cut-geometry trade-off.
- **4 corrupted rows** discovered via cross-column consistency checks (`z == carat`, `y == depth` — values that
  landed in the wrong column during data entry).
- Grouping by `price_per_carat` (removing the carat confound) makes the `clarity`/`cut`/`color` grade effects
  monotonic and confirms the Simpson's-paradox explanation for the raw non-monotonicity above.
- `cut`, `color`, `clarity` are **largely non-redundant** with each other (low mutual correlation) — safe to
  keep all three as separate model features.
- `volume = x * y * z` correlates with `carat` at **r ≈ 0.98** — foreshadows the feature-engineering decision to
  collapse the three raw dimensions into one engineered feature.

**Estimator selection & heuristic model:**

- A **log-linear OLS benchmark** (`log(price) ~ log(carat) + cut + color + clarity`, fit with `numpy.linalg.lstsq`)
  reaches **R² ≈ 0.90, MAPE ≈ 9.73%** on a held-out test split — establishing that tree-based / non-linear
  models are likely to do even better, and that a log-price target transform is worth carrying forward.
- A **heuristic pricing model** (`price ≈ carat × base_rate(carat_bin) × clarity_mult × color_mult × cut_mult`,
  with every rate/multiplier estimated from the training data) reaches **R² ≈ 0.83, MAPE ≈ 12.51%** — well above
  a carat-only linear fit (R² ≈ 0.61, MAPE ≈ 25.63%) and used throughout `5-models` as the baseline every ML
  model must beat.

### 4 · `notebooks/4-feat_eng` — Feature engineering

[`01-gmg-basic-feature-engineering-pipeline-2026_08_18.ipynb`](notebooks/4-feat_eng/01-gmg-basic-feature-engineering-pipeline-2026_08_18.ipynb)

- **Deduplication**: 250 exact duplicate rows found and dropped (19,231 → 18,981) — not flagged in the EDA, and
  a real leakage risk if a duplicate landed on both sides of a train/test split.
- **Careful outlier policy** (evidence-based, not a blanket IQR sweep):
    - Drops only the 4 EDA-identified **corrupted** rows (`z == carat`, `y == depth`) → **18,977 rows**, saved to
    `data/04_feature/diamantes_clean.parquet`.
    - A conservative 3×IQR check flags 38 `carat` outliers (0.2%) and 117 `depth` outliers (0.6%) — both are
    **deliberately kept**: the `carat` outliers are genuine large, valuable stones, and **98% (115/117) of the
    `depth` outliers are `Fair`-cut diamonds** — i.e. that unusual depth *is* the physical signature of a `Fair`
    cut, not noise to be removed.
- **Feature pipeline** (a `ColumnTransformer` reused unchanged by every notebook in `5-models`):
    - `carat`, `depth`, `table` → median-imputed + standard-scaled.
    - `x`, `y`, `z` → collapsed into one engineered `volume = x*y*z` feature (imputed + scaled) instead of passed
    raw, directly addressing the r ≈ 0.84–0.98 multicollinearity found in the EDA.
    - `cut`, `color`, `clarity` → **ordinal-encoded** (not one-hot) with an explicit worst → best category order
    (`color` encoded `J → D`, i.e. reversed from alphabetical, so a higher code always means a higher price
    premium, consistent with `cut`/`clarity`).
- **Train/test split**: stratified on a combined `clarity × cut` key (not a plain random split, and not just
  `clarity`), since some grade combinations are rare and a plain split risks leaving them out of one side
  entirely.
- **Recommendation carried into `5-models`**: model `log(price)` via `TransformedTargetRegressor`, and prefer a
  tree-based gradient-boosting estimator.

### 5 · `notebooks/5-models` — Model training, tuning & validation

Five notebooks plus two equivalent standalone scripts (`train_model_pipeline.py`,
`train_model_pipeline_hardtype.py`), all consuming `data/04_feature/diamantes_clean.parquet` and the feature
pipeline from stage 4.

| # | Notebook | What it establishes |
| --- | --- | --- |
| 01 | [`base_model`](notebooks/5-models/01-gmg-base_model-2026_08_18.ipynb) | A `scikit-learn`-compatible `HeuristicPriceRegressor` (same design as the EDA's heuristic, now a real estimator with a `fit`/`predict` API). **Test MAPE ≈ 12.47%, R² ≈ 0.836** — the baseline every model below must beat. |
| 02 | [`basic_algorithms_model_selection`](notebooks/5-models/02-gmg-basic_algorithms_model_selection-2026_08_18.ipynb) | Compares 8 regression algorithms (linear/Ridge/Lasso, KNN, Decision Tree, Random Forest, Gradient Boosting, Hist Gradient Boosting), all on `log(price)`. Linear models plateau at R² ≈ 0.70 (non-linear feature interactions dominate); `random_forest` is discarded for overfitting (train R² 0.993 vs. test 0.949); **`HistGradientBoostingRegressor` wins** after tuning: **MAPE ≈ 6.84%, R² ≈ 0.952** (vs. `GradientBoostingRegressor` runner-up at MAPE ≈ 6.90%). |
| 03 | [`first_model`](notebooks/5-models/03-gmg-first_model-2026_08_18.ipynb) | Clean, re-runnable training script for the winning model. **Saves the final artifact to `data/06_models/diamantes_price-hist_gradient_boosting-v1.joblib`.** |
| 04 | [`experiment-track-model`](notebooks/5-models/04-gmg-experiment-track-model-2026_08_18.ipynb) | Same pipeline, tracked with **MLflow** (experiment `diamantes_price_models`, local SQLite backend). Also saves an MLflow-format copy to `data/06_models/diamantes_price-hist_gradient_boosting-mlflow/`. |
| 05 | [`deepcheck-ML-process`](notebooks/5-models/05-gmg-deepcheck-ML-process-2026_08_18.ipynb) | Runs Deepchecks' data-integrity, train/test-validation and model-evaluation suites (32 checks). **30 pass**; the 2 failures are both already-understood: the raw `carat`/`x`/`y`/`z` multicollinearity (expected — mitigated by the `volume` feature) and a 10.79% train→test RMSE gap (mild overfitting, consistent with notebook 02's learning curve). Confirms the model is sound to treat as the project's production candidate. |

**Final model**, ready to load from either notebook:

```python
from joblib import load

model = load("data/06_models/diamantes_price-hist_gradient_boosting-v1.joblib")
predictions = model.predict(new_diamonds_df)  # columns: carat, depth, table, x, y, z, cut, color, clarity
```

### 6 · `notebooks/6-interpretation` — Model interpretation

[`01-gmg-modeloX_interpretation-2026_08_18.ipynb`](notebooks/6-interpretation/01-gmg-modeloX_interpretation-2026_08_18.ipynb)

A deep, multi-technique interpretation of the final model — permutation importance (3 metrics), SHAP (global
importance, beeswarm, dependence plots, pairwise interaction values), Partial Dependence/ICE (1D and a manual 2D
grid), per-diamond local explanations, and segment/error analysis — aimed at extracting checkable knowledge and
explicitly testing it against every earlier notebook's conclusions.

**Headline findings:**

| Finding | Detail |
|---|---|
| `clarity` outranks `carat` in permutation importance | Not a contradiction of the EDA's r ≈ 0.79 carat correlation — `carat`'s size signal is redundantly split across 4 correlated raw columns (`carat`, `x`, `y`, `z`), diluting each one's individual importance, while `clarity` has no redundant backup anywhere in the feature set. On the engineered features, SHAP correctly ranks the consolidated `volume` feature #1 — the `4-feat_eng` multicollinearity fix, vindicated quantitatively. |
| `depth`/`table` have a real effect the EDA missed | Both show a symmetric, non-monotonic "sweet spot" (inverted-V) around the average proportion in their SHAP dependence plots — invisible to Pearson correlation (r ≈ −0.07 / −0.03) precisely *because* it's non-monotonic and averages out to ≈0. |
| The heuristic model's independence assumption doesn't fully hold | SHAP interaction values show `color` × `clarity` as the strongest pairwise interaction in the whole model; a manual carat × clarity grid shows the IF-vs-I1 price gap grows from **164% to 228%** between 1.0 and 1.5 carat. `cut`, by contrast, barely interacts with anything — independence roughly holds there. |
| A real, double-confirmed anomaly | `cut = Premium` prices slightly *below* `cut = Very Good`, confirmed by two independent methods (SHAP dependence on the transformed ordinal codes, and a corrected reading of `PartialDependenceDisplay`'s categorical output, which orders bars alphabetically rather than worst→best — a plotting pitfall caught and documented rather than misreported). |
| Three weak segments, and they compound | Stones **>2.5ct** (14.4% MAPE vs. ~6.3% for the 1.0–1.5ct bulk of the data), **`Fair` cut** (11.5% MAPE — ties back to `4-feat_eng`'s finding that 98% of depth outliers are `Fair`-cut), and **`I1`/`SI2` clarity** (10.8% / 8.2% MAPE). The single worst mispredicted diamond in the test set sits at the intersection of all three. |
| An extrapolation trap, caught rather than reported at face value | A synthetic 0.5-carat query returns a suspiciously flat prediction, nearly identical to a 1.0-carat one — the training data's carat floor is exactly 1.00 (confirmed in `2-exploration`), so this is a tree-model boundary artifact, not a real "size barely matters below 1ct" finding. |

Closes with a point-by-point "Does the model agree with domain knowledge?" table comparing every finding above
against the `3-analysis`/`4-feat_eng`/`5-models` conclusions, and flags fixing the model artifact's picklability
(`compute_volume`/`volume_feature_names_out` currently must be redefined by every consumer script — see
`## ⚙️ Notable environment/dependency decisions` below) as a prerequisite for `7-deploy`.

### 7 · `notebooks/7-deploy` — Streamlit deployment

Two Streamlit apps expose the final model (`data/06_models/diamantes_price-hist_gradient_boosting-v1.joblib`)
as an interactive tool, fulfilling the `1-data` problem framing's goal of reducing seller/buyer information
asymmetry:

- [`diamantes-streamlit.py`](notebooks/7-deploy/diamantes-streamlit.py) — single-diamond price predictor: enter
  a diamond's `carat`/`depth`/`table`/`x`/`y`/`z`/`cut`/`color`/`clarity`, get a predicted price plus a
  **"Why this price?" breakdown** — the same `exp(shap_i)` multiplicative SHAP decomposition built in
  `6-interpretation`, showing which features pushed the price up or down and by how much.
- [`diamantes-streamlit-batch.py`](notebooks/7-deploy/diamantes-streamlit-batch.py) — the same single-diamond
  predictor as one tab, plus a **batch tab**: upload a CSV of many diamonds, get predictions (and average price)
  for all of them, with a downloadable results CSV. Column names/values are matched case-insensitively (e.g.
  `"ideal"`, `"IDEAL"` and `"Ideal"` all resolve to the `cut` grade `"Ideal"`).

Both apps warn explicitly when a submitted `carat < 1.00`, since the training data's carat floor is exactly
1.00 (see `2-exploration`) and predictions below that are extrapolations, per the `6-interpretation` finding.

**Run either app locally:**

```bash
uv run streamlit run notebooks/7-deploy/diamantes-streamlit.py
# or, for the batch-upload variant:
uv run streamlit run notebooks/7-deploy/diamantes-streamlit-batch.py
```

This starts a local server (by default at <http://localhost:8501>) and opens it in your browser. Both scripts
resolve the model path relative to their own location, so they can be run from any working directory. Stop the
server with `Ctrl+C`.

## 🗺️ Roadmap

Stage `8-reports` has not been started yet.

## ⚙️ Notable environment/dependency decisions

- `scikit-learn`, `mlflow`, `deepchecks`, `kaleido==0.2.1`, `shap` and `streamlit` were added as project
  dependencies to support the `5-models`, `6-interpretation` and `7-deploy` notebooks (see `pyproject.toml`).
  `kaleido` is pinned below 1.0 because `kaleido>=1.0` requires a separately installed Chrome/Chromium binary
  that isn't available in this environment.
- `notebooks/5-models/05-...deepcheck...ipynb` documents and works around two upstream Deepchecks 0.19.1 bugs
  against this project's pinned `numpy>=2.0` / `scikit-learn>=1.9` (a removed `np.Inf` alias and a renamed
  `'max_error'` scorer) with small, narrowly-scoped compatibility shims.
- `notebooks/5-models/04-...experiment-track-model...ipynb` uses MLflow's `cloudpickle` serialization format
  instead of the new default `skops`, because `skops` refuses to (de)serialize the notebook-local `volume`
  feature-engineering function as an "untrusted type".
- **Resolved fragility**: `joblib.dump` pickles the `volume` `FunctionTransformer`'s `compute_volume` /
  `volume_feature_names_out` functions *by reference*, not by value. Early on this meant `__main__.compute_volume`
  — every script/notebook loading the model had to redefine both functions locally before calling `joblib.load()`.
  Both functions now live in the importable `src/data/features.py` module, and
  `data/06_models/diamantes_price-hist_gradient_boosting-v1.joblib` was retrained/re-saved through
  `pipelines.training_pipeline.run` so it pickles a stable `data.features.compute_volume` reference instead — see
  `src/README.md` for the current library structure and usage.

## Credits

This project was generated from [@JoseRZapata]'s [data science project template] and completed as coursework
for the Módulo de Ciencia de datos en producción.

---
[@JoseRZapata]: https://github.com/JoseRZapata
[Cruft]: https://cruft.github.io/cruft/
[data science project template]: https://github.com/JoseRZapata/data-science-project-template
[Data structure]: data/README.md
