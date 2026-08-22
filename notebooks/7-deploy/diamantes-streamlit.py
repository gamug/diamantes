import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# make `src` importable (this script runs standalone via `streamlit run`, outside pytest's
# `pythonpath = ["src"]` config — see CLAUDE.md for that convention)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.constants import FEATURE_COLUMNS, ORDINAL_CATEGORIES
from inference.explain import explain_prediction
from inference.predict import (
    is_out_of_training_range,
    load_production_model,
    predict_price,
)

# https://docs.streamlit.io/library/api-reference

# HOW TO RUN THE APP:
# uv run streamlit run notebooks/7-deploy/diamantes-streamlit.py


def get_user_data() -> pd.DataFrame:
    """
    Get a single diamond's physical/quality characteristics from the user, as a one-row DataFrame
    matching the model's expected input columns.

    :return: one-row DataFrame with columns carat, depth, table, x, y, z, cut, color, clarity
    """
    user_data = {}

    col_a, col_b = st.columns(2)
    with col_a:
        user_data["carat"] = st.number_input(
            "Carat (weight):", min_value=0.2, max_value=5.5, value=1.0, step=0.01
        )
        user_data["depth"] = st.number_input(
            "Depth (%):", min_value=40.0, max_value=80.0, value=61.8, step=0.1
        )
        user_data["table"] = st.number_input(
            "Table (%):", min_value=40.0, max_value=100.0, value=57.0, step=0.1
        )
    with col_b:
        user_data["x"] = st.number_input(
            "Length x (mm):", min_value=0.0, max_value=12.0, value=6.4, step=0.01
        )
        user_data["y"] = st.number_input(
            "Width y (mm):", min_value=0.0, max_value=12.0, value=6.4, step=0.01
        )
        user_data["z"] = st.number_input(
            "Depth z (mm):", min_value=0.0, max_value=10.0, value=4.0, step=0.01
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        user_data["cut"] = st.select_slider(
            "Cut:", options=ORDINAL_CATEGORIES["cut"], value="Ideal"
        )
    with col2:
        user_data["color"] = st.select_slider(
            "Color (J worst -> D best):", options=ORDINAL_CATEGORIES["color"], value="G"
        )
    with col3:
        user_data["clarity"] = st.select_slider(
            "Clarity (I1 worst -> IF best):", options=ORDINAL_CATEGORIES["clarity"], value="SI1"
        )

    return pd.DataFrame([user_data])[FEATURE_COLUMNS]


@st.cache_resource
def get_model(model_dir: str) -> object:
    """
    Loads the trained diamond-price model in joblib format.

    Args:
        model_dir (str): The directory where the trained model artifact is stored.

    Returns:
        object: the fitted pipeline (preprocessing + log-price target transform
        + HistGradientBoostingRegressor), see notebooks/5-models/03-gmg-first_model-2026_08_18.ipynb.
    """
    with st.spinner("Loading model..."):
        model = load_production_model(Path(model_dir))

    return model


def main() -> None:
    st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎")

    # title
    st.header(body="💎 Diamond Price Predictor")
    st.write(
        "Estimate the market price of a diamond from its physical and quality characteristics, using the "
        "model trained in `notebooks/5-models/03-gmg-first_model-2026_08_18.ipynb` "
        "(MAPE ≈ 6.84%, R² ≈ 0.952 on held-out test data)."
    )

    # choose the trained model you want to use to make predictions
    project_path = Path(__file__).resolve().parents[2]
    model_dir = project_path / "data" / "06_models"

    # get the data from the user
    diamond = get_user_data()

    # load the model and predict the price for the given diamond
    model = get_model(model_dir=str(model_dir))

    if st.button("Predict price", type="primary"):
        predicted_price = predict_price(model, diamond)[0]

        st.write("")
        st.metric(label="Predicted price", value=f"${predicted_price:,.2f}")

        if is_out_of_training_range(diamond).iloc[0]:
            st.warning(
                "⚠️ The training data only contains diamonds with carat ≥ 1.00 "
                "(see `notebooks/2-exploration`). This prediction is an extrapolation and may be unreliable."
            )

        st.subheader("Why this price?")
        st.dataframe(explain_prediction(model, diamond), hide_index=True)


if __name__ == "__main__":
    main()
