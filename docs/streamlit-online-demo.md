# Online demo — Streamlit app (issue #37)

A single-page web form for **online inference**: enter one diamond's
characteristics, get an estimated price from the model trained by
`src/training_pipeline.py` (issue #24).

- **App:** [`src/streamlit_app.py`](../src/streamlit_app.py)
- **Model:** `data/06_models/diamantes_price-hist_gradient_boosting-v1.joblib` (committed)
- **Live app:** _add the Streamlit Community Cloud URL here after deploying_

## Usage instructions (end users)

1. Open the app URL.
2. Fill in the form:
   - **Carat** (weight, 0.2–5.01), **Depth %** (43–79), **Table %** (43–95).
   - **x / y / z** — length, width and depth in millimetres.
   - **Cut / Color / Clarity** — pick a grade (dropdowns, worst → best).
   - The fields are pre-filled with a typical mid-market stone, and each numeric
     field is clamped to the range documented in `data/01_raw/datos_diamantes_Info.txt`,
     so an out-of-range value cannot be submitted.
3. Press **Estimate price**.
4. Read the result:
   - **Estimated price** — the model's point estimate, in USD.
   - **Rough range** — `estimate ± 7.4 %` (the model's held-out test MAPE). This
     is a scale reference, *not* a calibrated prediction interval.
   - Any **⚠ warnings** about inputs the model handles poorly (see Limitations).
   - Expand **Model input** to see the row after cleaning + feature engineering
     (`volume = x·y·z`), i.e. exactly what the model received.

## Run it locally

```bash
make install_env                       # once
uv run streamlit run src/streamlit_app.py
```

Opens on <http://localhost:8501>. If the model file is missing, the app says so —
run `uv run python src/training_pipeline.py` first.

## Deployment — Streamlit Community Cloud

1. Push this repo to GitHub (already done) with the model artifact committed at
   `data/06_models/diamantes_price-hist_gradient_boosting-v1.joblib`.
2. Sign in at <https://share.streamlit.io> with the GitHub account that owns the repo.
3. **New app → From existing repo** and set:
   - **Repository:** `gamug/diamantes`
   - **Branch:** `main`
   - **Main file path:** `src/streamlit_app.py`
   - **Python version:** 3.12 (Advanced settings)
4. Streamlit Cloud installs from [`requirements.txt`](../requirements.txt) — a
   minimal, pinned subset of the project dependencies (the app needs only
   `streamlit`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `joblib`; it does
   **not** import `deepchecks` / `matplotlib`). `streamlit run` puts `src/` on
   `sys.path`, so the flat `src/` modules import by bare name.
5. **Deploy.** First build takes a few minutes; afterwards the app is reachable at
   a stable `https://<subdomain>.streamlit.app` URL. Paste that URL into
   **Live app** above and into the PR / issue #37.

Redeploys are automatic on every push to `main`.

## Limitations & expected behaviour

- **Carat < 1.0 ct is extrapolation.** The training data over-represents larger
  stones (its carat floor is ~1.0 ct — see `notebooks/2-exploration`), so
  predictions below that are unreliable; the app shows a warning.
- **Geometry consistency.** `depth` is *defined* as `2·z / (x + y) · 100`. If the
  entered `depth` disagrees with the `x/y/z` you typed by more than 2 percentage
  points, the app warns you to re-check the dimensions (it still predicts).
- **Point estimate, rough band.** The ± band is the model's average test MAPE
  applied symmetrically — it is not a per-prediction confidence interval.
- **Static model.** The dataset is a one-time course snapshot; the model is not
  retrained on new data. Retraining means re-running `src/training_pipeline.py`
  and committing the new `.joblib`.
- **Online only.** Batch scoring (CSV upload) is out of scope for issue #37 —
  use `uv run python src/inference_pipeline.py` for that.

## Evidence of correct operation

Local run of the app's prediction path against the committed model
(`PYTHONPATH=src uv run python -c "import streamlit_app as a; ..."`):

| Input | Estimate | Rough range | Warning |
| --- | --- | --- | --- |
| pristine form defaults (1.0 ct · Ideal · G · VS2 · 61.8/57 · 6.40/6.42/3.97) | **$6,310** | $5,843 – $6,777 | — |
| 2.0 ct · Premium · F · VS2 · 62.0/58 · 8.10/8.05/5.00 | **$17,893** | $16,562 – $19,223 | — |
| 0.5 ct · Fair · J · I1 · 64.0/58 · 5.00/5.00/3.20 | **$836** | $774 – $898 | carat 0.50 is below 1.0 ct — extrapolation |

`streamlit run src/streamlit_app.py --server.headless true` starts cleanly and
`/_stcore/health` returns `ok` (HTTP 200).

_Add screenshots of the deployed app (form + a successful prediction) here._
