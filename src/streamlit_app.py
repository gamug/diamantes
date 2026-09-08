"""Streamlit demo: diamond price prediction, online and in batch (issues #37, #38).

Two tabs, one deployed app:

* **One stone** — a web form for **online inference**: one diamond in, one
  estimated price out (issue #37).
* **A file of stones** — **batch inference** (issue #38): upload a CSV with
  many rows, every row is priced, and the table of predictions can be
  downloaded or viewed in place.

Both tabs are served by the model trained in `training_pipeline.py` (issue #24,
loaded from `data/06_models/`) and reuse the exact feature transformation the
batch `inference_pipeline.py` applies (`diamond_features.build_inference_features`:
type-fix, range / category validation, `volume = x*y*z`, imputer-friendly
`NaN`s). The app does **not** import `inference_pipeline` itself, so the deployed
footprint stays small (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `scipy`,
`joblib` — no `deepchecks` / `matplotlib`); the batch view charts with native
`st.scatter_chart`.

The layout follows a grading report: a two-column ledger (the stone's
attributes | the assessed value) under a midnight display-velvet hero band,
on a deep blue-black ground (dark mode). The estimated figure and a single
spectral "fire" ray beneath it are the one loud element.

Run locally:

    uv run streamlit run src/streamlit_app.py

Deploy: see `docs/streamlit-online-demo.md` (online) and
`docs/streamlit-batch-demo.md` (batch).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.base import BaseEstimator

from diamond_features import (
    KNOWN_CATEGORIES,
    MEASUREMENT_BOUNDS,
    build_inference_features,
    implied_depth_percentage,
)

#: Repo root: this file lives at ``src/streamlit_app.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
#: Trained model artifact (committed so the deployed app has it).
MODEL_PATH = REPO_ROOT / "data" / "06_models" / "diamantes_price-hist_gradient_boosting-v1.joblib"
#: Written by ``training_pipeline.py``; may be absent on a fresh checkout.
METRICS_PATH = REPO_ROOT / "data" / "08_reporting" / "training_metrics.json"
#: Example batch-inference input, committed so the deployed app can offer it
#: as a download and reviewers can reproduce the batch view (issue #38).
SAMPLE_BATCH_CSV_PATH = REPO_ROOT / "data" / "01_raw" / "diamantes_batch_sample.csv"

#: Model input columns (mirrors ``training_pipeline.MODEL_FEATURES``; the fitted
#: ``ColumnTransformer`` selects by name, so only their presence matters).
MODEL_FEATURES: list[str] = ["carat", "depth", "table", "volume", "cut", "color", "clarity"]
#: Raw form fields, in the raw-CSV column order. Also the columns an uploaded
#: batch file must carry (``price`` is optional and ignored for inference).
RAW_INPUT_COLUMNS: list[str] = [
    "carat",
    "cut",
    "color",
    "clarity",
    "depth",
    "table",
    "x",
    "y",
    "z",
]
#: Column appended to the batch table with the model's estimate (mirrors
#: ``inference_pipeline.PREDICTION_COLUMN``).
PREDICTION_COLUMN = "predicted_price"

#: Fallback when ``training_metrics.json`` is not in the checkout.
DEFAULT_TEST_MAPE_PCT: float = 7.4
#: The training data over-represents >= 1 ct stones (see notebooks/2-exploration).
CARAT_EXTRAPOLATION_FLOOR: float = 1.0
#: Warn if the entered depth disagrees with the x/y/z geometry by more than this.
GEOMETRY_WARN_TOLERANCE_PP: float = 2.0

#: Sensible mid-market defaults for the form: a ~1 ct round-brilliant Ideal
#: stone whose dimensions are consistent with the entered depth, so the
#: pristine form triggers neither the extrapolation nor the geometry warning.
_DEFAULTS: dict[str, float | str] = {
    "carat": 1.0,
    "cut": "Ideal",
    "color": "G",
    "clarity": "VS2",
    "depth": 61.8,
    "table": 57.0,
    "x": 6.40,
    "y": 6.42,
    "z": 3.97,
}
_DECIMALS: dict[str, int] = {"carat": 2, "depth": 1, "table": 1, "x": 2, "y": 2, "z": 2}


def load_test_mape(path: Path = METRICS_PATH, default: float = DEFAULT_TEST_MAPE_PCT) -> float:
    """Read the model's held-out test MAPE (%), or fall back to ``default``."""
    try:
        return float(json.loads(path.read_text())["mape_pct"])
    except (OSError, ValueError, KeyError, TypeError):
        return default


def build_input_row(fields: dict[str, float | str]) -> pd.DataFrame:
    """One raw-shaped row (text cells, like a CSV line) for the transformation.

    Args:
        fields: One non-empty value per name in :data:`RAW_INPUT_COLUMNS`.

    Raises:
        ValueError: If a required raw column is absent, ``None`` or blank -- an
            incomplete input must be rejected, not silently imputed and scored.
    """
    missing = [
        column
        for column in RAW_INPUT_COLUMNS
        if fields.get(column) is None or str(fields.get(column)).strip() == ""
    ]
    if missing:
        raise ValueError(f"missing or empty input fields: {missing}")
    return pd.DataFrame(
        [{column: str(fields[column]) for column in RAW_INPUT_COLUMNS}], columns=RAW_INPUT_COLUMNS
    )


def prepare_features(raw_row: pd.DataFrame) -> pd.DataFrame:
    """Raw form row -> the model-input matrix (same transform as inference_pipeline)."""
    return build_inference_features(raw_row)[MODEL_FEATURES]


def predict_price(model: BaseEstimator, raw_row: pd.DataFrame) -> float:
    """Predicted price (USD) for a single raw form row."""
    return float(model.predict(prepare_features(raw_row))[0])


def price_interval(price: float, mape_pct: float) -> tuple[float, float]:
    """A rough ``price ± mape_pct%`` band (not a calibrated prediction interval)."""
    delta = price * mape_pct / 100.0
    return price - delta, price + delta


def input_warnings(*, carat: float, depth: float, x: float, y: float, z: float) -> list[str]:
    """Plain-language notes about inputs the model handles poorly or inconsistently."""
    warnings: list[str] = []
    if carat < CARAT_EXTRAPOLATION_FLOOR:
        warnings.append(
            f"This model learned mostly from stones of {CARAT_EXTRAPOLATION_FLOOR:.0f} carat and up. "
            f"At {carat:.2f} ct the estimate is a rough extrapolation."
        )
    if x > 0.0 and y > 0.0:
        implied = float(implied_depth_percentage(pd.DataFrame([{"x": x, "y": y, "z": z}])).iloc[0])
        if abs(depth - implied) > GEOMETRY_WARN_TOLERANCE_PP:
            warnings.append(
                f"The depth you entered ({depth:.1f}%) doesn't match what x, y and z imply "
                f"({implied:.1f}%). Check the measurements."
            )
    return warnings


# --- batch inference (issue #38) -----------------------------------
# Same transform + model as the online tab, applied to a whole uploaded file
# instead of one form row. Row-preserving: every input row gets a prediction
# (bad / missing cells are left for the model pipeline's imputers), mirroring
# ``inference_pipeline.run_inference_pipeline``.


def read_batch_csv(source: object) -> pd.DataFrame:
    """Read an uploaded batch file as text (no type coercion), like ``load_raw_data``.

    Args:
        source: Anything :func:`pandas.read_csv` accepts -- a path, a file-like
            buffer, or Streamlit's ``UploadedFile``.

    Returns:
        The raw table, every column ``object`` so
        :func:`diamond_features.build_inference_features` sees the original
        string values.
    """
    return pd.read_csv(source, dtype="object", low_memory=False)


def missing_required_columns(df: pd.DataFrame) -> list[str]:
    """Names from :data:`RAW_INPUT_COLUMNS` absent from ``df`` (order preserved)."""
    return [column for column in RAW_INPUT_COLUMNS if column not in df.columns]


def batch_predict(model: BaseEstimator, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Price every row of a raw batch table.

    Args:
        model: The fitted price model.
        raw_df: New data with the raw column layout (see
            :data:`RAW_INPUT_COLUMNS`), values as text.

    Returns:
        ``raw_df`` (index reset) with a float :data:`PREDICTION_COLUMN` column
        appended -- one prediction per input row, same order. Any other input
        columns are carried through untouched.

    Raises:
        ValueError: If a required raw column is missing, the file has no data
            rows, an input column already uses the reserved
            :data:`PREDICTION_COLUMN` name, or a model feature has no usable
            value anywhere in the file (which would make the fitted pipeline
            drop that column and mispredict every row).
    """
    missing = missing_required_columns(raw_df)
    if missing:
        raise ValueError(f"the file is missing required columns: {missing}")
    if PREDICTION_COLUMN in raw_df.columns:
        raise ValueError(f"an input column uses the reserved output name: {PREDICTION_COLUMN}")
    if raw_df.empty:
        raise ValueError("the file has no data rows")

    features = build_inference_features(raw_df)[MODEL_FEATURES]
    all_null = [column for column in features.columns if features[column].isna().all()]
    if all_null:
        raise ValueError(f"these model features have no usable value in the whole file: {all_null}")

    predictions = np.asarray(model.predict(features), dtype=float)
    result = raw_df.reset_index(drop=True).copy()
    result[PREDICTION_COLUMN] = predictions
    return result


def batch_summary(result: pd.DataFrame) -> dict[str, float]:
    """Mean / min / max of the predicted prices (NaNs ignored)."""
    prices = pd.to_numeric(result[PREDICTION_COLUMN], errors="coerce")
    return {
        "mean": float(prices.mean()),
        "min": float(prices.min()),
        "max": float(prices.max()),
    }


def batch_chart_data(result: pd.DataFrame) -> pd.DataFrame:
    """A numeric ``carat`` / :data:`PREDICTION_COLUMN` frame for ``st.scatter_chart``."""
    return pd.DataFrame(
        {
            "carat": pd.to_numeric(result["carat"], errors="coerce"),
            PREDICTION_COLUMN: pd.to_numeric(result[PREDICTION_COLUMN], errors="coerce"),
        }
    ).dropna()


def load_sample_batch_csv(path: Path = SAMPLE_BATCH_CSV_PATH) -> str | None:
    """Text of the committed example batch file, or ``None`` if it isn't there."""
    try:
        return path.read_text()
    except OSError:
        return None


# --- visual design ---------------------------------------------------
# Grounded in the grading-report vernacular: a two-column ledger (the stone's
# attributes | the assessed value), a deep blue-black ground (dark mode), a
# midnight display-velvet hero band, and a single spectral "fire" ray -- the
# one loud element -- under the estimated figure. Space Grotesk carries the
# number (it is the product); IBM Plex Sans carries everything else. Colour is
# left to Streamlit's own dark theme wherever a widget already reads well
# (the primary submit button, dropdown popovers), so the injected CSS stays
# thin and never fights the base theme -- see .streamlit/config.toml.

_FACET_MARK = """
<svg class="dpx-mark" viewBox="0 0 64 64" role="img" aria-label="Round brilliant, top view">
  <circle cx="32" cy="32" r="30"/>
  <polygon points="32,12 46,18 52,32 46,46 32,52 18,46 12,32 18,18"/>
  <polygon points="32,20 40,23 44,32 40,41 32,44 24,41 20,32 24,23"/>
  <path d="M32 12V20M46 18L40 23M52 32H44M46 46L40 41M32 52V44M18 46L24 41M12 32H20M18 18L24 23"/>
</svg>
"""

_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

:root{
  --ground:#0B131E; --panel:#152232; --panel-2:#1B2B3F; --line:#2A3A4E;
  --velvet:#101F33; --velvet-edge:#28405C;
  --ink:#EBF1F8; --slate:#B4C2D3; --graphite:#8291A5;
  --sans:'IBM Plex Sans',ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --figure:'Space Grotesk','IBM Plex Sans',ui-sans-serif,system-ui,sans-serif;
  --prism:linear-gradient(93deg,#79D2FF 0%,#B7A6FF 52%,#FFB1C6 100%);
}

/* keep Streamlit's Material (ligature) icons rendering as glyphs -- a broad
   font-family override used to leak into them and print the raw ligature name
   (e.g. "arrow_right") on top of the label. */
[data-testid="stIconMaterial"],
span[class*="material-symbols"], span[class*="material-icons"]{
  font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
}

[data-testid="stAppViewContainer"]{background:var(--ground);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"]{right:.5rem;}
.stApp{font-family:var(--sans);color:var(--ink);}
.stApp p, .stApp li, .stApp label{font-family:var(--sans);color:var(--ink);}
.block-container{max-width:1140px;padding-top:1.4rem;padding-bottom:3rem;}
.stApp [data-testid="stMainBlockContainer"]{padding-top:1.4rem;}

/* hero band -------------------------------------------------------- */
.dpx-hero{
  background:linear-gradient(150deg,var(--velvet) 0%,#0A1723 100%);
  color:#fff;border:1px solid var(--velvet-edge);border-radius:14px;
  padding:2.1rem 2.3rem;display:flex;gap:1.5rem;align-items:flex-start;
  margin:.2rem 0 1.9rem;
}
.dpx-mark{width:60px;height:60px;flex:none;margin-top:.15rem;
  fill:none;stroke:rgba(190,220,255,.5);stroke-width:1;}
.dpx-hero h1{font-family:var(--figure);font-weight:600;font-size:2rem;
  letter-spacing:-.015em;line-height:1.05;margin:0 0 .5rem;color:#fff;}
.dpx-hero p{font-family:var(--sans);font-size:1.02rem;line-height:1.5;
  color:rgba(220,230,242,.72);margin:0;max-width:52ch;}

/* section titles = report headers -------------------------------- */
.dpx-h{font-family:var(--figure);font-weight:600;font-size:1.02rem;
  color:var(--slate);margin:.1rem 0 1rem;padding-bottom:.55rem;
  border-bottom:1px solid var(--line);}
.dpx-cap{font-family:var(--sans);font-size:.82rem;color:var(--graphite);
  margin:1rem 0 .35rem;}

/* the ledger centre rule: the second column of the top split ----- */
[data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-of-type(2){
  border-left:1px solid var(--line);padding-left:2.4rem;
}
@media (max-width:820px){
  [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-of-type(2){
    border-left:0;padding-left:0;
  }
}

/* form fields --------------------------------------------------- */
[data-testid="stForm"]{border:0;padding:0;}
.stApp [data-testid="stWidgetLabel"] p{font-size:.86rem;color:var(--graphite);
  font-weight:500;margin-bottom:.25rem;}
[data-testid="stNumberInput"] input,
[data-baseweb="select"] > div{
  background:var(--panel);border:1px solid var(--line);border-radius:7px;
  color:var(--ink);font-variant-numeric:tabular-nums;
}
[data-testid="stNumberInput"] input:focus,
[data-baseweb="select"] > div:focus-within{
  border-color:#79D2FF;box-shadow:0 0 0 3px rgba(121,210,255,.16);outline:none;
}
/* number-input +/- steppers */
[data-testid="stNumberInput"] button{
  background:var(--panel);border:1px solid var(--line);color:var(--slate);
}
[data-testid="stNumberInput"] button:hover{background:var(--panel-2);color:var(--ink);}

/* submit button: shape + type only -- colour is the (dark) base theme's
   primary style, which already reads well and keeps the label legible. */
[data-testid="stFormSubmitButton"] button{
  font-family:var(--figure);font-weight:600;border-radius:8px;
  padding:.62rem 1rem;margin-top:.4rem;
  transition:filter .12s ease,transform .06s ease;
}
[data-testid="stFormSubmitButton"] button:hover{filter:brightness(1.08);}
[data-testid="stFormSubmitButton"] button:active{transform:translateY(1px);}

/* assessed value: the one loud element ------------------------- */
.dpx-empty{font-family:var(--sans);color:var(--graphite);font-size:.98rem;
  line-height:1.55;max-width:44ch;margin:.4rem 0 0;}
.dpx-value__label{display:block;font-family:var(--sans);font-size:.9rem;
  color:var(--graphite);margin-bottom:.35rem;}
.dpx-value__figure{display:block;font-family:var(--figure);font-weight:600;
  font-size:clamp(2.7rem,7vw,4.4rem);line-height:1;letter-spacing:-.02em;
  color:var(--ink);font-variant-numeric:tabular-nums;}
.dpx-value__ray{display:block;height:3px;width:min(100%,320px);border-radius:2px;
  margin:.9rem 0 1.05rem;background:var(--prism);transform-origin:left;
  box-shadow:0 0 18px rgba(121,210,255,.22);
  animation:dpx-draw .62s cubic-bezier(.2,.7,.2,1) both;}
.dpx-value__range{display:block;font-family:var(--sans);font-size:1.02rem;
  color:var(--slate);}
.dpx-value__note{font-family:var(--sans);font-size:.85rem;color:var(--graphite);
  line-height:1.5;max-width:46ch;margin:.45rem 0 0;}
@keyframes dpx-draw{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@media (prefers-reduced-motion:reduce){.dpx-value__ray{animation:none;}}

/* input notes: asides marked by a rule, not a card ------------- */
.dpx-notes{list-style:none;padding:0;margin:1.35rem 0 0;}
.dpx-notes li{border-left:2px solid var(--graphite);padding-left:.85rem;
  color:var(--slate);font-size:.9rem;line-height:1.5;margin:.55rem 0;}

/* expander + dataframe ---------------------------------------- */
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:8px;
  background:var(--panel);margin-top:1.1rem;}
[data-testid="stExpander"] summary p{font-family:var(--sans);color:var(--slate);
  font-size:.9rem;}
[data-testid="stExpander"] summary:hover p{color:var(--ink);}
[data-testid="stDataFrame"]{font-variant-numeric:tabular-nums;}

.dpx-foot{font-family:var(--sans);font-size:.82rem;color:var(--graphite);
  margin:2.4rem 0 0;padding-top:1rem;border-top:1px solid var(--line);}
</style>
"""


@st.cache_resource(show_spinner="Reading the model…")
def _load_model(path_str: str) -> BaseEstimator:  # pragma: no cover
    return load(path_str)


def _number_field(key: str) -> float:  # pragma: no cover
    decimals = _DECIMALS[key]
    labels = {
        "carat": "Carat",
        "depth": "Depth %",
        "table": "Table %",
        "x": "x",
        "y": "y",
        "z": "z",
    }
    # Only a floor of 0 (nothing physical is negative) -- deliberately no upper
    # bound and no documented-minimum clamp, so a value outside MEASUREMENT_BOUNDS
    # *can* be entered and the out-of-range confirmation dialog can catch it
    # (st.number_input would otherwise silently clamp it away). See
    # `range_warnings` / `_render_online_tab`.
    return float(
        st.number_input(
            labels[key],
            min_value=0.0,
            value=float(_DEFAULTS[key]),
            step=10.0**-decimals,
            format=f"%.{decimals}f",
        )
    )


def _select_field(key: str) -> str:  # pragma: no cover
    options = KNOWN_CATEGORIES[key]
    return str(st.selectbox(key.capitalize(), options, index=options.index(str(_DEFAULTS[key]))))


def _render_hero() -> None:  # pragma: no cover
    st.html(
        f'<div class="dpx-hero">{_FACET_MARK}'
        "<div><h1>Diamond price</h1>"
        "<p>Estimate what a stone is worth from its grade and measurements.</p></div></div>"
    )


def _render_estimate(
    price: float, low: float, high: float, mape_pct: float
) -> None:  # pragma: no cover
    st.html(
        '<div class="dpx-value">'
        '<span class="dpx-value__label">Estimated market value</span>'
        f'<span class="dpx-value__figure">${price:,.0f}</span>'
        '<span class="dpx-value__ray" aria-hidden="true"></span>'
        f'<span class="dpx-value__range">Likely between ${low:,.0f} and ${high:,.0f}</span>'
        f'<p class="dpx-value__note">The range spans the model\'s average error on held-out '
        f"stones (±{mape_pct:.1f}%). It's a guide, not a guaranteed price.</p></div>"
    )


def _render_notes(notes: list[str]) -> None:  # pragma: no cover
    if not notes:
        return
    items = "".join(f"<li>{note}</li>" for note in notes)
    st.html(f'<ul class="dpx-notes">{items}</ul>')


# --- online tab: out-of-range confirmation ------------------------
# The numeric fields accept values outside their documented MEASUREMENT_BOUNDS
# (see _number_field), and a value can also be inside those bounds yet outside
# the range the model was *trained* on (sub-1 ct, or a depth that contradicts
# x/y/z). Rather than price such inputs silently and only footnote the problem,
# the tab pops a modal: "Estimate anyway" runs the model, "Go back" holds the
# result until the inputs are fixed or resubmitted.

#: Session-state keys for the confirmation handshake.
_ONLINE_PENDING = "online_pending_fields"
_ONLINE_CONFIRMED = "online_confirmed_signature"
#: Stored in ``_ONLINE_CONFIRMED`` when the user declines the modal.
_ONLINE_CANCELLED = "__cancelled__"

#: Human labels for the numeric fields (used in range messages).
_FIELD_LABELS: dict[str, str] = {"carat": "Carat", "depth": "Depth %", "table": "Table %"}


def fields_signature(fields: dict[str, float | str]) -> str:
    """Order-independent string identity of a set of form inputs."""
    return json.dumps({key: str(value) for key, value in fields.items()}, sort_keys=True)


def range_warnings(fields: dict[str, float | str]) -> list[str]:
    """Notes for any numeric field sitting outside its documented physical range."""
    out: list[str] = []
    for key, (low, high) in MEASUREMENT_BOUNDS.items():
        raw = fields.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        value = float(raw)
        if not low <= value <= high:
            label = _FIELD_LABELS.get(key, key)
            out.append(f"{label} {value:g} is outside the supported {low:g} to {high:g} range.")
    return out


def submission_warnings(fields: dict[str, float | str]) -> list[str]:
    """Every reason the given inputs are outside the model's reliable range."""
    return range_warnings(fields) + input_warnings(
        carat=float(fields["carat"]),
        depth=float(fields["depth"]),
        x=float(fields["x"]),
        y=float(fields["y"]),
        z=float(fields["z"]),
    )


@st.dialog("Inputs outside the model's reliable range")
def _confirm_out_of_range(notes: list[str], signature: str) -> None:  # pragma: no cover
    st.markdown(
        "The model can still return a number, but one or more inputs sit outside "
        "the range it was trained on, so the estimate would be unreliable:"
    )
    for note in notes:
        st.markdown(f"- {note}")
    go, back = st.columns(2)
    if go.button("Estimate anyway", type="primary", width="stretch", key="oor_go"):
        st.session_state[_ONLINE_CONFIRMED] = signature
        st.rerun()
    if back.button("Go back", width="stretch", key="oor_back"):
        st.session_state[_ONLINE_CONFIRMED] = _ONLINE_CANCELLED
        st.rerun()


def _stone_form() -> tuple[dict[str, float | str], bool]:  # pragma: no cover
    """The left-hand input form; returns the entered fields and the submit state."""
    st.markdown('<h2 class="dpx-h">The stone</h2>', unsafe_allow_html=True)
    fields: dict[str, float | str] = {}
    left, right = st.columns(2)
    with left:
        fields["carat"] = _number_field("carat")
        fields["color"] = _select_field("color")
        fields["depth"] = _number_field("depth")
    with right:
        fields["cut"] = _select_field("cut")
        fields["clarity"] = _select_field("clarity")
        fields["table"] = _number_field("table")
    st.markdown('<p class="dpx-cap">Measurements (mm)</p>', unsafe_allow_html=True)
    mx, my, mz = st.columns(3)
    with mx:
        fields["x"] = _number_field("x")
    with my:
        fields["y"] = _number_field("y")
    with mz:
        fields["z"] = _number_field("z")
    submitted = st.form_submit_button("Assess value", type="primary", width="stretch")
    return fields, submitted


_EMPTY_PROMPT = (
    "<p class=\"dpx-empty\">Enter a stone's details and assess it. You'll get an "
    "estimated market value with a likely range.</p>"
)
_AWAITING_PROMPT = (
    '<p class="dpx-empty">These inputs are outside the model\'s reliable range. '
    "Confirm in the dialog to run the estimate anyway.</p>"
)
_DECLINED_PROMPT = (
    '<p class="dpx-empty">Estimate held back — the inputs are outside the model\'s '
    "reliable range. Adjust them, or press <strong>Assess value</strong> again to confirm.</p>"
)


def _render_online_value(
    model: BaseEstimator,
    test_mape: float,
    pending: dict[str, float | str] | None,
    notes: list[str],
    state: str,
) -> None:  # pragma: no cover
    """The right-hand panel: a prompt, the confirmation notice, or the estimate."""
    st.markdown('<h2 class="dpx-h">Assessed value</h2>', unsafe_allow_html=True)
    prompts = {"empty": _EMPTY_PROMPT, "awaiting": _AWAITING_PROMPT, "declined": _DECLINED_PROMPT}
    if state in prompts:
        st.markdown(prompts[state], unsafe_allow_html=True)
        return

    assert pending is not None  # state == "ready" only when a submission is pending
    raw_row = build_input_row(pending)
    price = predict_price(model, raw_row)
    low, high = price_interval(price, test_mape)
    _render_estimate(price, low, high, test_mape)
    _render_notes(notes)
    with st.expander("What the model used"):
        model_input = prepare_features(raw_row).iloc[0]
        st.dataframe(
            pd.DataFrame(
                {"feature": model_input.index, "value": model_input.astype(str).to_numpy()}
            ),
            hide_index=True,
            width="stretch",
        )


def _render_online_tab(model: BaseEstimator, test_mape: float) -> None:  # pragma: no cover
    """One-stone form on the left, the assessed value (with an out-of-range gate) on the right."""
    stone_col, value_col = st.columns([1.04, 0.96], gap="large")
    with stone_col, st.form("stone", border=False):
        fields, submitted = _stone_form()

    if submitted:
        st.session_state[_ONLINE_PENDING] = dict(fields)
        st.session_state.pop(_ONLINE_CONFIRMED, None)  # a new submission must be re-confirmed

    pending: dict[str, float | str] | None = st.session_state.get(_ONLINE_PENDING)
    notes = submission_warnings(pending) if pending else []
    signature = fields_signature(pending) if pending else ""
    confirmed = st.session_state.get(_ONLINE_CONFIRMED)
    awaiting_confirmation = bool(notes) and confirmed not in (signature, _ONLINE_CANCELLED)

    if not pending:
        state = "empty"
    elif awaiting_confirmation:
        state = "awaiting"
    elif notes and confirmed == _ONLINE_CANCELLED:
        state = "declined"
    else:
        state = "ready"

    with value_col:
        _render_online_value(model, test_mape, pending, notes, state)

    if awaiting_confirmation:
        _confirm_out_of_range(notes, signature)


def _render_batch_tab(model: BaseEstimator) -> None:  # pragma: no cover
    """Upload a CSV of many stones, price every row, view / download the table."""
    st.markdown('<h2 class="dpx-h">A file of stones</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="dpx-empty">Upload a CSV with one row per diamond and the columns '
        "<code>carat, cut, color, clarity, depth, table, x, y, z</code>. Every row is priced; "
        "download the results or read them below.</p>",
        unsafe_allow_html=True,
    )

    sample = load_sample_batch_csv()
    if sample is not None:
        st.download_button(
            "Download a sample CSV",
            sample,
            file_name="diamantes_batch_sample.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded is None:
        return

    try:
        raw_df = read_batch_csv(uploaded)
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Couldn't read that file as CSV: {exc}")
        return

    missing = missing_required_columns(raw_df)
    if missing:
        st.error(
            f"The file is missing required columns: {missing}. "
            f"It needs all of: {', '.join(RAW_INPUT_COLUMNS)}."
        )
        return

    try:
        result = batch_predict(model, raw_df)
    except ValueError as exc:
        st.error(f"Can't price this file: {exc}")
        return
    summary = batch_summary(result)

    st.markdown(f'<p class="dpx-cap">Priced {len(result):,} diamonds</p>', unsafe_allow_html=True)
    mean_col, min_col, max_col = st.columns(3)
    mean_col.metric("Average", f"${summary['mean']:,.0f}")
    min_col.metric("Lowest", f"${summary['min']:,.0f}")
    max_col.metric("Highest", f"${summary['max']:,.0f}")

    st.dataframe(result, hide_index=True, width="stretch")

    chart_data = batch_chart_data(result)
    if not chart_data.empty:
        st.markdown('<p class="dpx-cap">Predicted price vs. carat</p>', unsafe_allow_html=True)
        st.scatter_chart(chart_data, x="carat", y=PREDICTION_COLUMN, height=280)

    st.download_button(
        "Download predictions (CSV)",
        result.to_csv(index=False),
        file_name="diamantes_predictions.csv",
        mime="text/csv",
        type="primary",
    )


def main() -> None:  # pragma: no cover
    """Render the demo page: an online tab and a batch tab over one model."""
    st.set_page_config(page_title="Diamond price", page_icon="\U0001f48e", layout="wide")
    st.markdown(_STYLE, unsafe_allow_html=True)
    _render_hero()

    if not MODEL_PATH.is_file():
        st.error(
            "The model file isn't here yet. Run `uv run python src/training_pipeline.py` to build it."
        )
        st.stop()

    model = _load_model(str(MODEL_PATH))
    test_mape = load_test_mape()

    online_tab, batch_tab = st.tabs(["One stone", "A file of stones"])
    with online_tab:
        _render_online_tab(model, test_mape)
    with batch_tab:
        _render_batch_tab(model)

    st.markdown(
        '<p class="dpx-foot">Trained on a course dataset of about 53,000 graded stones. '
        "Price one stone online, or a whole file at once.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
