"""Streamlit online demo: single-diamond price prediction (issue #37).

A web form for **online inference** — one diamond in, one estimated price out —
served by the model trained in `training_pipeline.py` (issue #24) and loaded
from `data/06_models/`.

It reuses the exact feature transformation the batch `inference_pipeline.py`
applies (`diamond_features.build_inference_features`: type-fix, range / category
validation, `volume = x*y*z`, imputer-friendly `NaN`s), but does **not** import
`inference_pipeline` itself, so the deployed app's dependency footprint stays
small (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib` — no `deepchecks`
/ `matplotlib`).

The layout follows a grading report: a two-column ledger (the stone's
attributes | the assessed value) under a midnight display-velvet hero band,
on a cool "colorless-stone" ground. The estimated figure and a single
spectral "fire" ray beneath it are the one loud element.

Run locally:

    uv run streamlit run src/streamlit_app.py

Deploy: see `docs/streamlit-online-demo.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

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

#: Model input columns (mirrors ``training_pipeline.MODEL_FEATURES``; the fitted
#: ``ColumnTransformer`` selects by name, so only their presence matters).
MODEL_FEATURES: list[str] = ["carat", "depth", "table", "volume", "cut", "color", "clarity"]
#: Raw form fields, in the raw-CSV column order.
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


# --- visual design ---------------------------------------------------
# Grounded in the grading-report vernacular: a two-column ledger (the stone's
# attributes | the assessed value), a "colorless-stone" cool-white ground, a
# midnight display-velvet hero band, and a single spectral "fire" ray -- the
# one loud element -- under the estimated figure. Space Grotesk carries the
# number (it is the product); IBM Plex Sans carries everything else.

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
  --colorless:#F5F7FB; --mist:#ECEFF5; --line:#D7DDE7;
  --velvet:#0E1B2C; --velvet-2:#16283E;
  --ink:#1B242E; --slate:#3C4551; --graphite:#5C6470;
  --sans:'IBM Plex Sans',ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --figure:'Space Grotesk','IBM Plex Sans',ui-sans-serif,system-ui,sans-serif;
  --prism:linear-gradient(93deg,#79D2FF 0%,#B7A6FF 52%,#FFB1C6 100%);
}

[data-testid="stAppViewContainer"]{background:var(--colorless);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"]{right:.5rem;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span{font-family:var(--sans);color:var(--ink);}
.block-container{max-width:1140px;padding-top:1.4rem;padding-bottom:3rem;}
.stApp [data-testid="stMainBlockContainer"]{padding-top:1.4rem;}

/* hero band -------------------------------------------------------- */
.dpx-hero{
  background:var(--velvet);color:#fff;border-radius:14px;
  padding:2.1rem 2.3rem;display:flex;gap:1.5rem;align-items:flex-start;
  margin:.2rem 0 1.9rem;
}
.dpx-mark{width:60px;height:60px;flex:none;margin-top:.15rem;
  fill:none;stroke:rgba(255,255,255,.55);stroke-width:1;}
.dpx-hero h1{font-family:var(--figure);font-weight:600;font-size:2rem;
  letter-spacing:-.015em;line-height:1.05;margin:0 0 .5rem;color:#fff;}
.dpx-hero p{font-family:var(--sans);font-size:1.02rem;line-height:1.5;
  color:rgba(255,255,255,.72);margin:0;max-width:52ch;}

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
  background:var(--mist);border:1px solid var(--line);border-radius:7px;
  font-variant-numeric:tabular-nums;
}
[data-testid="stNumberInput"] input:focus,
[data-baseweb="select"] > div:focus-within{
  border-color:var(--velvet);box-shadow:0 0 0 3px rgba(14,27,44,.13);outline:none;
}
[data-testid="stFormSubmitButton"] button{
  font-family:var(--figure);font-weight:600;border-radius:7px;border:0;
  background:var(--velvet);color:#fff;padding:.6rem 1rem;margin-top:.4rem;
  transition:background .12s ease,transform .06s ease;
}
[data-testid="stFormSubmitButton"] button:hover{background:var(--velvet-2);}
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
[data-testid="stExpander"] summary{font-family:var(--sans);color:var(--graphite);
  font-size:.9rem;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:8px;
  background:transparent;margin-top:1.1rem;}
[data-testid="stDataFrame"]{font-variant-numeric:tabular-nums;}

.dpx-foot{font-family:var(--sans);font-size:.82rem;color:var(--graphite);
  margin:2.4rem 0 0;padding-top:1rem;border-top:1px solid var(--line);}
</style>
"""


@st.cache_resource(show_spinner="Reading the model…")
def _load_model(path_str: str) -> BaseEstimator:  # pragma: no cover
    return load(path_str)


def _number_field(key: str) -> float:  # pragma: no cover
    low, high = MEASUREMENT_BOUNDS[key]
    decimals = _DECIMALS[key]
    labels = {
        "carat": "Carat",
        "depth": "Depth %",
        "table": "Table %",
        "x": "x",
        "y": "y",
        "z": "z",
    }
    return float(
        st.number_input(
            labels[key],
            min_value=float(low),
            max_value=float(high),
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


def main() -> None:  # pragma: no cover
    """Render the online-demo page."""
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

    stone_col, value_col = st.columns([1.04, 0.96], gap="large")

    with stone_col, st.form("stone", border=False):
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

    with value_col:
        st.markdown('<h2 class="dpx-h">Assessed value</h2>', unsafe_allow_html=True)
        if not submitted:
            st.markdown(
                "<p class=\"dpx-empty\">Enter a stone's details and assess it. You'll get an "
                "estimated market value with a likely range.</p>",
                unsafe_allow_html=True,
            )
        else:
            notes = input_warnings(
                carat=float(fields["carat"]),
                depth=float(fields["depth"]),
                x=float(fields["x"]),
                y=float(fields["y"]),
                z=float(fields["z"]),
            )
            raw_row = build_input_row(fields)
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

    st.markdown(
        '<p class="dpx-foot">Trained on a course dataset of about 53,000 graded stones. '
        "Online estimates for one stone at a time.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
