# Train pipeline model - diamond price regression
# 2026-08-21

# ## 📚 Import  libraries

# %%
# base libraries for data science
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler

# ## 💾 Load data
# Use a more robust way to find the data directory relative to this script,
# consistent with notebooks/4-feat_eng and notebooks/5-models/*.ipynb.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

dataset = pd.read_parquet(DATA_DIR / "04_feature/diamantes_clean.parquet", engine="pyarrow")

# selected features (per notebooks/4-feat_eng/01-gmg-basic-feature-engineering-pipeline-2026_08_18.ipynb)
numeric_features = ["carat", "depth", "table"]
volume_source_features = ["x", "y", "z"]
categorical_ordinal_features = ["cut", "color", "clarity"]
target = "price"

selected_features = numeric_features + volume_source_features + categorical_ordinal_features
dataset_features = dataset[[*selected_features, target]]

# ## Convert data types

dataset_features[numeric_features + volume_source_features] = dataset_features[
    numeric_features + volume_source_features
].astype("float32")

dataset_features[categorical_ordinal_features] = dataset_features[
    categorical_ordinal_features
].astype("category")

# ### target variable
dataset_features[target] = dataset_features[target].astype("int64")

# data/04_feature/diamantes_clean.parquet is already deduplicated and free of the corrupted rows identified in
# the EDA (z == carat, y == depth data-entry swaps); that cleaning happened upstream in the
# feature-engineering pipeline, so it is not repeated here.

# 👨‍🏭 Feature Engineering
# volume = x*y*z replaces the raw x/y/z dimensions (they are near-collinear with carat, r up to 0.98 per the
# EDA); cut/color/clarity are ordinal-encoded worst -> best so a higher code always means a higher price premium.

ordinal_categories = {
    "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    "color": ["J", "I", "H", "G", "F", "E", "D"],
    "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
}


def compute_volume(X: np.ndarray) -> np.ndarray:
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return (x * y * z).reshape(-1, 1)


def volume_feature_names_out(
    transformer: FunctionTransformer, input_features: list[str]
) -> np.ndarray:
    """Named (picklable) replacement for a lambda: FunctionTransformer always outputs 1 column, 'volume'."""
    return np.array(["volume"])


volume_transformer = FunctionTransformer(compute_volume, feature_names_out=volume_feature_names_out)

numeric_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

volume_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("volume", volume_transformer),
        ("scaler", StandardScaler()),
    ]
)

categorical_ord_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ordinal",
            OrdinalEncoder(
                categories=[ordinal_categories[c] for c in categorical_ordinal_features]
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_features),
        ("volume", volume_pipe, volume_source_features),
        ("cat_ordinal", categorical_ord_pipe, categorical_ordinal_features),
    ]
)

# Train / Test split

X_features = dataset_features[selected_features]
Y_target = dataset_features[target]

# stratify on clarity x cut, consistent with the feature-engineering notebook: some grade combinations
# are rare, so a plain random split risks under/over-representing them between train and test.
clarity_cut_key = (
    dataset_features["clarity"].astype(str) + "_" + dataset_features["cut"].astype(str)
)

x_train, x_test, y_train, y_test = train_test_split(
    X_features, Y_target, stratify=clarity_cut_key, test_size=0.2, random_state=42
)

# Create pipeline
# log-price target transform (linearizes the convex carat -> price relationship, guarantees positive
# predictions), per notebooks/4-feat_eng's recommendation.

data_model_pipeline = TransformedTargetRegressor(
    regressor=Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor(random_state=42)),
        ]
    ),
    func=np.log,
    inverse_func=np.exp,
)

# Hyperparameter tunning
# model / grid selected in notebooks/5-models/02-gmg-basic_algorithms_model_selection-2026_08_18.ipynb and
# confirmed in notebooks/5-models/03-gmg-first_model-2026_08_18.ipynb.
score = "neg_mean_absolute_percentage_error"

hyperparameters = {
    "regressor__model__max_iter": [200, 300, 400],
    "regressor__model__max_depth": [6, 10, None],
    "regressor__model__learning_rate": [0.03, 0.05, 0.1],
    "regressor__model__l2_regularization": [0.0, 1.0],
}

grid_search = GridSearchCV(
    data_model_pipeline,
    hyperparameters,
    cv=5,
    scoring=score,
    n_jobs=8,
)
grid_search.fit(x_train, y_train)

best_data_model_pipeline = grid_search.best_estimator_

# evaluation
y_pred = best_data_model_pipeline.predict(x_test)

metric_result = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"evaluation metric (MAPE %): {metric_result}")

# Save the model
# %%
DATA_MODEL = Path(__file__).resolve().parents[2] / "data" / "06_models"
DATA_MODEL.mkdir(parents=True, exist_ok=True)
# %%
dump(
    best_data_model_pipeline,
    DATA_MODEL / "diamantes_price-hist_gradient_boosting-v1.joblib",
    protocol=5,
)
