# Batch demo — Streamlit app (issue #38)

**Batch inference**: upload a CSV with many diamonds, get a price for every row,
then download the table or read it in the app. It is the second tab of the same
Streamlit app that serves the [online demo](./streamlit-online-demo.md) — one
deployment, one model, two modes.

- **App:** [`src/streamlit_app.py`](../src/streamlit_app.py) — tab **“A file of stones”**
- **Model:** `data/06_models/diamantes_price-hist_gradient_boosting-v1.joblib` (committed)
- **Example input:** [`data/01_raw/diamantes_batch_sample.csv`](../data/01_raw/diamantes_batch_sample.csv) (18 rows)
- **Example output:** [`data/07_model_output/diamantes_batch_sample_predictions.csv`](../data/07_model_output/diamantes_batch_sample_predictions.csv)
- **Live app:** <https://diamantes-model.streamlit.app/> — open the **“A file of stones”** tab (same deployment as the online demo)

## How batch mode works

The batch tab runs the **same transformation and model as the online tab** and
as the standalone [`src/inference_pipeline.py`](../src/inference_pipeline.py):

1. The upload is read as text (`read_batch_csv`, `dtype="object"`) — no silent
   type coercion, exactly like `feature_pipeline.load_raw_data`.
2. Required columns are checked (`missing_required_columns`): the file must have
   `carat, cut, color, clarity, depth, table, x, y, z`. A `price` column is
   allowed and ignored; any other extra columns are carried through to the
   output untouched. The file is also rejected (with an `st.error`, nothing
   predicted) if it has **no data rows**, if an input column already uses the
   reserved name **`predicted_price`**, or if a whole model feature is unusable
   across every row (e.g. every `carat` unparseable) — that last case would
   otherwise make the fitted pipeline drop the column and mis-score the batch.
3. `diamond_features.build_inference_features` fixes types, nulls out
   out-of-range / unknown-category values, and engineers `volume = x·y·z`.
   **Rows are never dropped** — one prediction per input row; bad or missing
   cells are left as `NaN` for the model pipeline's imputers.
4. `model.predict` prices every row; the estimate is appended as
   `predicted_price`.
5. The app shows a summary (count, average / lowest / highest price), the full
   predictions table, a *predicted price vs. carat* scatter, and a **Download
   predictions (CSV)** button.

## Usage instructions (end users)

1. Open the app URL and click the **“A file of stones”** tab.
2. *(optional)* Click **Download a sample CSV** to get a ready-to-use input file
   (`data/01_raw/diamantes_batch_sample.csv`).
3. Click **Browse files** and choose a `.csv` with one diamond per row and the
   columns listed above (order does not matter; a header row is required).
4. The app prices every row and shows:
   - **Priced N diamonds** with **Average / Lowest / Highest** price metrics.
   - The **predictions table** — every input column plus `predicted_price`.
   - **Predicted price vs. carat** — a scatter of the batch.
5. Click **Download predictions (CSV)** to save the table as
   `diamantes_predictions.csv`.

### Preparing an input file

- Column names must match exactly (lower-case): `carat, cut, color, clarity,
  depth, table, x, y, z`.
- `cut` ∈ {Fair, Good, Very Good, Premium, Ideal}; `color` ∈ {D…J};
  `clarity` ∈ {I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF}. Unknown grades are
  treated as missing and imputed.
- Numeric ranges (values outside are treated as missing): `carat` 0.2–5.01,
  `depth` 43–79, `table` 43–95, `x` ≤ 10.74, `y` ≤ 58.9, `z` ≤ 31.8.

## Run it locally

```bash
make install_env                       # once
uv run streamlit run src/streamlit_app.py
```

Opens on <http://localhost:8501>; use the **“A file of stones”** tab. The
standalone batch script (writes Parquet + a PNG under `data/07_model_output/`)
is still available too:

```bash
uv run python src/inference_pipeline.py
```

## Deployment — Streamlit Community Cloud

The batch tab needs **no separate deployment** — it is the same
`src/streamlit_app.py`, so the app already live at
<https://diamantes-model.streamlit.app/> gains the *“A file of stones”* tab on
the next push to `main`. The steps below are the full record of how that one
app was created (they apply equally to `#37` and `#38`).

### Prerequisites (already in the repo)

- `data/06_models/diamantes_price-hist_gradient_boosting-v1.joblib` — the trained
  model, committed (Streamlit Cloud has no build step to produce it).
- `data/01_raw/diamantes_batch_sample.csv` — the example input, committed; the
  app reads it from disk to power the *Download a sample CSV* button.
- `requirements.txt` at the repo root — the exact, pinned runtime deps Streamlit
  Cloud installs (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `scipy`,
  `joblib`). No `deepchecks` / `matplotlib`: the batch view charts with native
  `st.scatter_chart`.
- `.streamlit/config.toml` — the dark theme; picked up automatically.

### One-time setup on share.streamlit.io

1. Go to <https://share.streamlit.io> and **sign in with GitHub** using the
   account that owns `gamug/diamantes`; authorise the Streamlit Community Cloud
   app when GitHub prompts.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill the form with **exactly these values**:

   | Form field | Value to enter |
   | --- | --- |
   | **Repository** | `gamug/diamantes` |
   | **Branch** | `main` |
   | **Main file path** | `src/streamlit_app.py` |
   | **App URL** *(optional)* | a `diamantes-…` subdomain of your choice — this deployment uses `diamantes-model`, giving <https://diamantes-model.streamlit.app/> |

4. *(optional)* **Advanced settings…** → **Python version** `3.12`; leave
   **Secrets** empty — the app needs none.
5. Click **Deploy**. The first build runs `pip install -r requirements.txt` and
   takes a few minutes; when it finishes the app is live at the App URL from
   step 3. `streamlit run` puts `src/` on `sys.path`, so the flat `src/` modules
   import by bare name.

### After the first deploy

- **Redeploys are automatic** on every push to the `main` branch — no action
  needed for the batch tab beyond merging its PR.
- Manage / reboot / view logs from the app's **⋮ menu** on
  <https://share.streamlit.io>.
- The default upload cap is **200 MB per file** — far above the ~53 k-row course
  dataset; raise it with `[server] maxUploadSize` in `.streamlit/config.toml`
  only if ever needed.

## Limitations & expected behaviour

- **One prediction per row, always** — as long as the file itself is usable.
  Individual bad cells (unparseable numbers, unknown grades) are *not* rejected:
  they become `NaN`, the model's imputers fill them, and the row still gets a
  (less reliable) price. What *is* rejected up front, with nothing predicted: a
  file with no data rows, a `predicted_price` input column, and any model
  feature that is `NaN` for **every** row.
- **No prediction interval per row.** The batch view reports point estimates
  only. For the ± band, use the online tab (it applies the model's ±7.4 % test
  MAPE to a single stone).
- **Sub-1 ct stones are extrapolation.** The training data over-represents
  larger stones; predictions below ~1 ct are unreliable (the online tab warns
  about this; the batch view does not annotate individual rows).
- **Columns must be present.** A file missing any of the nine required columns
  is rejected with a message naming the missing ones — nothing is predicted.
- **Static model.** One-time course snapshot; retraining means re-running
  `src/training_pipeline.py` and committing the new `.joblib`.

## Evidence of correct operation

### Example input → output (reproducible)

`data/01_raw/diamantes_batch_sample.csv` (18 stones spanning 0.20–2.01 ct) run
through the batch path:

```text
$ PYTHONPATH=src uv run python -c "
import pandas as pd; from joblib import load; import streamlit_app as a
m = load(a.MODEL_PATH); raw = a.read_batch_csv(a.SAMPLE_BATCH_CSV_PATH)
res = a.batch_predict(m, raw)
res.to_csv(a.REPO_ROOT/'data/07_model_output/diamantes_batch_sample_predictions.csv', index=False)
print(a.batch_summary(res))"
{'mean': 4341.09, 'min': 405.66, 'max': 16621.89}
```

| carat | cut | color | clarity | … | predicted_price |
| ---: | --- | --- | --- | --- | ---: |
| 0.20 | Premium | D | VS2 | … | $406 |
| 0.51 | Ideal | G | VVS2 | … | $1,904 |
| 1.02 | Ideal | F | SI1 | … | $5,319 |
| 1.97 | Very Good | H | VS2 | … | $16,622 |
| 2.01 | Premium | I | SI1 | … | $15,208 |

Full table: [`data/07_model_output/diamantes_batch_sample_predictions.csv`](../data/07_model_output/diamantes_batch_sample_predictions.csv).

### Automated tests

`tests/test_streamlit_app.py` covers the batch path:

- **Pure helpers** — `missing_required_columns` (names absent columns in order),
  `read_batch_csv` (keeps every column as text), `batch_predict` (one row in →
  one `predicted_price` out, order preserved, extra columns carried, unparseable
  *cells* still scored, and an end-to-end run through a real
  `build_model_pipeline`; raises on a missing column, a header-only file, a
  reserved `predicted_price` input column, or a wholly-unusable feature),
  `batch_summary`, `batch_chart_data`, `load_sample_batch_csv`, and a round-trip
  of the committed sample file.
- **Rendered page (headless `AppTest`)** — the page exposes an *“One stone”* and
  an *“A file of stones”* tab; uploading the sample CSV produces a predictions
  dataframe with a `predicted_price` column and a *Download predictions (CSV)*
  button; a file missing columns surfaces a *“missing required columns”* error,
  and a header-only file a *“no data rows”* error (no crash).

### Deployed app

_Add screenshots here: the empty batch tab, a completed run (metrics + table +
scatter), and the downloaded `diamantes_predictions.csv` opened in a
spreadsheet._
