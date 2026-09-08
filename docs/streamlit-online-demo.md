# Online demo — Streamlit app (issue #37)

A web form for **online inference**: enter one diamond's characteristics, get an
estimated price from the model trained by `src/training_pipeline.py` (issue #24).
It is the **“One stone”** tab of `src/streamlit_app.py`; the **“A file of
stones”** tab does batch inference — see
[`streamlit-batch-demo.md`](./streamlit-batch-demo.md) (issue #38).

- **App:** [`src/streamlit_app.py`](../src/streamlit_app.py)
- **Model:** `data/06_models/diamantes_price-hist_gradient_boosting-v1.joblib` (committed)
- **Live app:** <https://diamantes-model.streamlit.app/> (**“One stone”** tab)

## Usage instructions (end users)

The page is a two-column ledger: **The stone** on the left, **Assessed value** on
the right.

1. Open the app URL.
2. Under **The stone**, fill in the form:
   - **Carat** (0.2–5.01), **Depth %** (43–79), **Table %** (43–95).
   - **Cut / Color / Clarity** — pick a grade (worst → best).
   - **Measurements (mm)** — `x`, `y`, `z` (length, width, depth).
   - Fields are pre-filled with a typical 1-carat round brilliant. Each numeric
     field is clamped to the range in `data/01_raw/datos_diamantes_Info.txt`, so an
     out-of-range value can't be submitted.
3. Press **Assess value**.
   - If any input is outside the range the model was trained on (sub-1 ct, a
     `depth` that contradicts `x/y/z`, or — should the bounds ever be relaxed —
     a value past its documented min/max), a dialog lists the problems and asks
     to confirm: **Estimate anyway** runs the model; **Go back** holds the
     result until you adjust the inputs (or press **Assess value** again to
     confirm).
4. Read the right panel:
   - **Estimated market value** — the point estimate, in USD.
   - **Likely between $X and $Y** — `estimate ± 7.4 %` (the model's held-out test
     MAPE). A guide to scale, *not* a calibrated prediction interval.
   - Any short **notes** (marked with a left rule) about inputs the model handles
     poorly — see Limitations.
   - Expand **What the model used** to see the row after cleaning and feature
     engineering (`volume = x·y·z`), i.e. exactly what the model received.

## Run it locally

```bash
make install_env                       # once
uv run streamlit run src/streamlit_app.py
```

Opens on <http://localhost:8501>. If the model file is missing, the app says so —
run `uv run python src/training_pipeline.py` first.

## Deployment — Streamlit Community Cloud

The app is deployed at <https://diamantes-model.streamlit.app/>. It was created
once, from the `main` branch, and redeploys automatically on every push.

1. Push this repo to GitHub (done) with the model artifact committed at
   `data/06_models/diamantes_price-hist_gradient_boosting-v1.joblib` — Streamlit
   Cloud has no build step to produce it.
2. Sign in at <https://share.streamlit.io> with the GitHub account that owns
   `gamug/diamantes`; authorise the Streamlit Community Cloud GitHub app.
3. Click **Create app** → **Deploy a public app from GitHub**, and fill the form
   with **exactly these values**:

   | Form field | Value to enter |
   | --- | --- |
   | **Repository** | `gamug/diamantes` |
   | **Branch** | `main` |
   | **Main file path** | `src/streamlit_app.py` |
   | **App URL** *(optional)* | a `diamantes-…` subdomain of your choice — this deployment uses `diamantes-model`, giving <https://diamantes-model.streamlit.app/> |

4. *(optional)* **Advanced settings…** → **Python version** `3.12`; leave
   **Secrets** empty — the app needs none.
5. Click **Deploy**. The first build installs
   [`requirements.txt`](../requirements.txt) — a minimal, pinned subset of the
   project deps (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `scipy`,
   `joblib`; **no** `deepchecks` / `matplotlib`). `streamlit run` puts `src/` on
   `sys.path`, so the flat `src/` modules import by bare name. First build takes
   a few minutes.

Redeploys are automatic on every push to `main`. The same field-by-field notes,
plus the batch tab, are in
[`streamlit-batch-demo.md`](./streamlit-batch-demo.md#deployment--streamlit-community-cloud).

## Limitations & expected behaviour

- **Out-of-range inputs are gated.** Sub-1 ct stones (the training data's carat
  floor is ~1.0 ct — see `notebooks/2-exploration`), a `depth` that disagrees
  with `2·z / (x + y) · 100` by more than 2 percentage points, or a value past
  its documented min/max all trigger a confirmation dialog before the model
  runs. The estimate is only produced after **Estimate anyway**; the same notes
  then stay visible beside the result. **Go back** withholds the estimate.
- **Confirmed out-of-range estimates are still unreliable.** Confirming past the
  dialog does not make the number trustworthy — it just records that you chose
  to see it anyway.
- **Point estimate, rough band.** The ± band is the model's average test MAPE
  applied symmetrically — it is not a per-prediction confidence interval.
- **Static model.** The dataset is a one-time course snapshot; the model is not
  retrained on new data. Retraining means re-running `src/training_pipeline.py`
  and committing the new `.joblib`.
- **This tab is one stone at a time.** Batch scoring (CSV upload) lives in the
  **“A file of stones”** tab — see
  [`streamlit-batch-demo.md`](./streamlit-batch-demo.md) — or run
  `uv run python src/inference_pipeline.py`.

## Design

The layout borrows the vernacular of a diamond **grading report**: a two-column
ledger — the stone's attributes on the left, the assessed value on the right,
separated by a single hairline — under a midnight display-velvet hero band. The
page runs in **dark mode**: a deep blue-black ground (`#0B131E`) with lifted
slate panels for the field fills, near-white ink, and a spectral cyan for the
primary action and focus rings. Two typefaces: **Space Grotesk** for the
estimated figure and section titles (the number is the product, so it gets the
characterful face), **IBM Plex Sans** for everything else. The one loud element
is the value figure and a single spectral "fire" ray beneath it (the dispersion
a diamond throws when rotated), which draws in once on **Assess value** and is
the page's only non-user-triggered motion (`prefers-reduced-motion` disables
it). The dark palette is set natively in `.streamlit/config.toml` so Streamlit's
own widgets (inputs, dropdowns, the submit button) inherit it; the bespoke
pieces live in the app's injected `<style>` (`.dpx-*`) and deliberately leave
colour to the base theme wherever it already reads well.

## Evidence of correct operation

Prediction path against the committed model (`PYTHONPATH=src uv run python -c
"import streamlit_app as a; ..."`):

| Input | Estimated market value | Likely range | Note |
| --- | --- | --- | --- |
| form defaults — 1.0 ct · Ideal · G · VS2 · 61.8/57 · 6.40/6.42/3.97 | **$6,310** | $5,843 – $6,777 | — |
| 2.0 ct · Premium · F · VS2 · 62.0/58 · 8.10/8.05/5.00 | **$17,893** | $16,562 – $19,223 | — |
| 0.5 ct · Fair · J · I1 · 64.0/58 · 5.00/5.00/3.20 | **$836** | $774 – $898 | sub-1 ct — confirm the dialog first |

Streamlit's headless `AppTest` (`tests/test_streamlit_app.py`) loads the page,
renders the 6 number + 3 select fields with no exception, submits the form, and
asserts the estimated figure and the "What the model used" table appear. For an
out-of-range submission (carat 0.5) it asserts the confirmation dialog opens
with **Estimate anyway** / **Go back**, that the figure is *not* shown until
**Estimate anyway** is clicked, and that **Go back** leaves the estimate held
back. `range_warnings` / `submission_warnings` / `fields_signature` are unit
tested directly. `streamlit run … --server.headless true` also starts cleanly
(`/_stcore/health` → `ok`).

*Add screenshots of the deployed app (the ledger + a completed assessment) here.*
