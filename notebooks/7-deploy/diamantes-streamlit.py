from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st
from joblib import load
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer

# https://docs.streamlit.io/library/api-reference

# HOW TO RUN THE APP:
# uv run streamlit run notebooks/7-deploy/diamantes-streamlit.py


# --- required to unpickle the trained model's `volume` feature transformer ---
# joblib pickles a FunctionTransformer's callables *by reference* (module + name), not by value,
# so any script that loads data/06_models/diamantes_price-hist_gradient_boosting-v1.joblib must
# redefine these two functions, with the exact same names, before calling `load()` below - see
# README.md's "Known fragility" note under "Notable environment/dependency decisions", and the
# same pattern in notebooks/5-models/{03,04,05} and notebooks/6-interpretation/01.
def compute_volume(X: np.ndarray) -> np.ndarray:
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return (x * y * z).reshape(-1, 1)


def volume_feature_names_out(
    transformer: FunctionTransformer, input_features: list[str]
) -> np.ndarray:
    """Named (picklable) replacement for a lambda: FunctionTransformer always outputs 1 column, 'volume'."""
    return np.array(["volume"])


# worst -> best, matching notebooks/4-feat_eng's OrdinalEncoder category order
ORDINAL_CATEGORIES = {
    "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    "color": ["J", "I", "H", "G", "F", "E", "D"],
    "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
}
FEATURE_COLUMNS = ["carat", "depth", "table", "x", "y", "z", "cut", "color", "clarity"]
MODEL_FILE_NAME = "diamantes_price-hist_gradient_boosting-v1.joblib"


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
def load_model(model_file_path: str) -> TransformedTargetRegressor:
    """
    Loads the trained diamond-price model in joblib format.

    Args:
        model_file_path (str): The path where the trained model is stored in joblib format.

    Returns:
        TransformedTargetRegressor: the fitted pipeline (preprocessing + log-price target transform
        + HistGradientBoostingRegressor), see notebooks/5-models/03-gmg-first_model-2026_08_18.ipynb.
    """

    with st.spinner("Loading model..."):
        model = load(model_file_path)

    return model


def explain_prediction(model: TransformedTargetRegressor, diamond: pd.DataFrame) -> None:
    """
    Show a SHAP-based, multiplicative breakdown of which features drove this prediction, following
    the same exp(shap_i) decomposition used in notebooks/6-interpretation/01-....ipynb.

    Args:
        model (TransformedTargetRegressor): the fitted model
        diamond (pd.DataFrame): one-row DataFrame with the diamond's characteristics
    """
    inner_pipeline = model.regressor_
    preprocessor = inner_pipeline.named_steps["preprocessor"]
    regressor = inner_pipeline.named_steps["model"]

    transformed = preprocessor.transform(diamond)
    explainer = shap.TreeExplainer(regressor)
    explanation = explainer(transformed)

    baseline = np.exp(explanation.base_values[0])
    st.caption(f"Baseline price for a typical diamond in the training data: ${baseline:,.2f}")

    contributions = pd.Series(explanation.values[0], index=preprocessor.get_feature_names_out())
    contributions = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    factor_df = pd.DataFrame(
        {
            "feature": contributions.index,
            "multiplier": [f"{np.exp(v):.3f}x" for v in contributions.to_numpy()],
            "effect": [
                "increases price" if v > 0 else "decreases price" for v in contributions.to_numpy()
            ],
        }
    )
    st.dataframe(factor_df, hide_index=True)


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
    model_file_path = project_path / "data" / "06_models" / MODEL_FILE_NAME

    # get the data from the user
    diamond = get_user_data()

    # load the model and predict the price for the given diamond
    model = load_model(model_file_path=str(model_file_path))

    if st.button("Predict price", type="primary"):
        predicted_price = model.predict(diamond)[0]

        st.write("")
        st.metric(label="Predicted price", value=f"${predicted_price:,.2f}")

        if diamond["carat"].iloc[0] < 1.0:
            st.warning(
                "⚠️ The training data only contains diamonds with carat ≥ 1.00 "
                "(see `notebooks/2-exploration`). This prediction is an extrapolation and may be unreliable."
            )

        st.subheader("Why this price?")
        explain_prediction(model, diamond)


if __name__ == "__main__":
    main()
