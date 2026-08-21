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
# uv run streamlit run notebooks/7-deploy/diamantes-streamlit-batch.py


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


def preprocess_batch_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a batch CSV's columns (case/format tolerant) to what the model expects.

    Args:
        df (pd.DataFrame): the original dataframe read from the uploaded CSV

    Returns:
        pd.DataFrame: normalized dataframe ready for prediction
    """
    processed_df = df.copy()
    processed_df.columns = [c.strip().lower() for c in processed_df.columns]

    for grade_col, categories in ORDINAL_CATEGORIES.items():
        if grade_col in processed_df.columns:
            lookup = {c.lower(): c for c in categories}
            processed_df[grade_col] = (
                processed_df[grade_col]
                .astype(str)
                .str.strip()
                .map(lambda v, lookup=lookup: lookup.get(v.lower(), v))
            )

    numeric_columns = ["carat", "depth", "table", "x", "y", "z"]
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

    return processed_df


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


def individual_prediction_tab(model: TransformedTargetRegressor) -> None:
    """
    Display the individual prediction interface and predict the price of a single diamond.

    Args:
        model (TransformedTargetRegressor): The trained model
    """
    diamond = get_user_data()

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


def batch_prediction_tab(model: TransformedTargetRegressor) -> None:
    """
    Display the batch prediction interface and predict prices for every diamond in a CSV file.

    Args:
        model (TransformedTargetRegressor): The trained model
    """
    st.subheader("Upload a CSV file with diamond characteristics")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            processed_df = preprocess_batch_data(df)
            missing_cols = [col for col in FEATURE_COLUMNS if col not in processed_df.columns]

            if missing_cols:
                st.warning(
                    f"Warning: your data is missing these columns: {', '.join(missing_cols)}"
                )
                st.info(f"Required columns: {', '.join(FEATURE_COLUMNS)}")
            elif st.button("Predict prices"):
                with st.spinner("Processing data and making predictions..."):
                    predictions = model.predict(processed_df[FEATURE_COLUMNS])

                    result_df = df.copy()
                    result_df["Predicted_Price"] = predictions

                    out_of_range = processed_df["carat"] < 1.0
                    if out_of_range.any():
                        st.warning(
                            f"⚠️ {int(out_of_range.sum())} row(s) have carat < 1.00 (outside the training "
                            "range, see `notebooks/2-exploration`) — those predictions are extrapolations."
                        )

                    st.success("Predictions completed!")
                    st.subheader("Prediction results")
                    st.dataframe(result_df)

                    st.metric("Average predicted price", f"${predictions.mean():,.2f}")

                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name="diamantes_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error processing the file: {e}")
            st.info("Please make sure your CSV file is properly formatted.")
    else:
        st.info("Please upload a CSV file with diamond characteristics.")

        st.subheader("Sample CSV format:")
        sample_data = pd.DataFrame(
            {
                "carat": [1.01, 1.51, 2.03],
                "cut": ["Ideal", "Premium", "Very Good"],
                "color": ["G", "H", "E"],
                "clarity": ["VS2", "SI1", "SI2"],
                "depth": [61.8, 62.0, 60.5],
                "table": [57.0, 58.0, 59.0],
                "x": [6.42, 7.32, 8.10],
                "y": [6.45, 7.28, 8.15],
                "z": [3.98, 4.53, 4.92],
            }
        )
        st.dataframe(sample_data)


def main() -> None:
    st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎")

    # title
    st.header(body="💎 Diamond Price Predictor")
    st.write(
        "Estimate the market price of one or many diamonds from their physical and quality characteristics, "
        "using the model trained in `notebooks/5-models/03-gmg-first_model-2026_08_18.ipynb` "
        "(MAPE ≈ 6.84%, R² ≈ 0.952 on held-out test data)."
    )

    # load the model
    project_path = Path(__file__).resolve().parents[2]
    model_file_path = project_path / "data" / "06_models" / MODEL_FILE_NAME
    model = load_model(model_file_path=str(model_file_path))

    tab1, tab2 = st.tabs(["Individual Prediction", "Batch Prediction"])

    with tab1:
        individual_prediction_tab(model)

    with tab2:
        batch_prediction_tab(model)


if __name__ == "__main__":
    main()
