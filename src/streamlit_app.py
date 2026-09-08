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
    """Non-blocking notes about inputs the model handles poorly or inconsistently."""
    warnings: list[str] = []
    if carat < CARAT_EXTRAPOLATION_FLOOR:
        warnings.append(
            f"carat {carat:.2f} is below {CARAT_EXTRAPOLATION_FLOOR:.1f} ct — the training data "
            "over-represents larger stones, so this prediction is an extrapolation."
        )
    if x > 0.0 and y > 0.0:
        implied = float(implied_depth_percentage(pd.DataFrame([{"x": x, "y": y, "z": z}])).iloc[0])
        if abs(depth - implied) > GEOMETRY_WARN_TOLERANCE_PP:
            warnings.append(
                f"the entered depth ({depth:.1f}%) disagrees with the depth implied by x/y/z "
                f"({implied:.1f}% = 2·z / (x + y) · 100) by more than "
                f"{GEOMETRY_WARN_TOLERANCE_PP:.0f} pp — double-check the dimensions."
            )
    return warnings


@st.cache_resource(show_spinner="Loading model…")
def _load_model(path_str: str) -> BaseEstimator:  # pragma: no cover
    return load(path_str)


def _number_field(key: str) -> float:  # pragma: no cover
    low, high = MEASUREMENT_BOUNDS[key]
    decimals = _DECIMALS[key]
    labels = {
        "carat": "Carat (weight)",
        "depth": "Depth (%)",
        "table": "Table (%)",
        "x": "x — length (mm)",
        "y": "y — width (mm)",
        "z": "z — depth (mm)",
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


def main() -> None:  # pragma: no cover
    """Render the online-demo page."""
    st.set_page_config(page_title="Diamond price — online demo", page_icon="💎", layout="centered")
    st.title("💎 Diamond price — online demo")
    st.caption(
        "Enter one diamond's characteristics to get a price estimate from the trained "
        "HistGradientBoostingRegressor (issue #24). **Online inference only.**"
    )

    if not MODEL_PATH.is_file():
        st.error(
            f"Model file not found at `{MODEL_PATH.relative_to(REPO_ROOT)}`. "
            "Run `uv run python src/training_pipeline.py` to create it."
        )
        st.stop()

    model = _load_model(str(MODEL_PATH))
    test_mape = load_test_mape()

    with st.form("diamond"):
        fields: dict[str, float | str] = {}
        col1, col2, col3 = st.columns(3)
        with col1:
            fields["carat"] = _number_field("carat")
            fields["cut"] = _select_field("cut")
        with col2:
            fields["depth"] = _number_field("depth")
            fields["color"] = _select_field("color")
        with col3:
            fields["table"] = _number_field("table")
            fields["clarity"] = _select_field("clarity")
        dim1, dim2, dim3 = st.columns(3)
        with dim1:
            fields["x"] = _number_field("x")
        with dim2:
            fields["y"] = _number_field("y")
        with dim3:
            fields["z"] = _number_field("z")
        submitted = st.form_submit_button("Estimate price", type="primary")

    if not submitted:
        st.info("Fill in the form and press **Estimate price**.")
        return

    for warning in input_warnings(
        carat=float(fields["carat"]),
        depth=float(fields["depth"]),
        x=float(fields["x"]),
        y=float(fields["y"]),
        z=float(fields["z"]),
    ):
        st.warning(warning)

    raw_row = build_input_row(fields)
    price = predict_price(model, raw_row)
    low, high = price_interval(price, test_mape)

    st.metric("Estimated price", f"${price:,.0f}")
    st.write(
        f"Rough range **${low:,.0f} - ${high:,.0f}** (+/-{test_mape:.1f} %, the model's held-out "
        "test MAPE - a scale reference, not a calibrated prediction interval)."
    )
    with st.expander("Model input (after cleaning + feature engineering)"):
        st.dataframe(prepare_features(raw_row).T.rename(columns={0: "value"}))


if __name__ == "__main__":
    main()
