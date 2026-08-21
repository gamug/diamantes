# Train pipeline model hardcode just for academic purposes - diamond price regression
# 2026-08-21

# ## 📚 Import  libraries

# base libraries for data science

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

# 💾 Load data

dataset = pd.read_parquet("data/04_feature/diamantes_clean.parquet")

dataset_features = dataset[
    [
        "carat",
        "depth",
        "table",
        "x",
        "y",
        "z",
        "cut",
        "color",
        "clarity",
        "price",
    ]
]

# Convert data types

# Numerical variables
dataset[["carat", "depth", "table", "x", "y", "z"]] = dataset[
    ["carat", "depth", "table", "x", "y", "z"]
].astype("float32")

# Categorical variables
dataset[["cut", "color", "clarity"]] = dataset[["cut", "color", "clarity"]].astype("category")

# ### target variable
dataset["price"] = dataset["price"].astype("int64")

# 👨‍🏭 Feature Engineering
# volume = x*y*z replaces the raw x/y/z dimensions (near-collinear with carat, r up to 0.98 per the EDA);
# cut/color/clarity are ordinal-encoded worst -> best.

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
                categories=[
                    ordinal_categories["cut"],
                    ordinal_categories["color"],
                    ordinal_categories["clarity"],
                ]
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, ["carat", "depth", "table"]),
        ("volume", volume_pipe, ["x", "y", "z"]),
        ("cat_ordinal", categorical_ord_pipe, ["cut", "color", "clarity"]),
    ]
)

# Train / Test split
X_features = dataset_features.drop("price", axis="columns")
Y_target = dataset_features["price"]

clarity_cut_key = (
    dataset_features["clarity"].astype(str) + "_" + dataset_features["cut"].astype(str)
)

x_train, x_test, y_train, y_test = train_test_split(
    X_features, Y_target, stratify=clarity_cut_key, test_size=0.2, random_state=42
)

# Create pipeline
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
    scoring="neg_mean_absolute_percentage_error",
    n_jobs=8,
)
grid_search.fit(x_train, y_train)

best_data_model_pipeline = grid_search.best_estimator_

# evaluation
y_pred = best_data_model_pipeline.predict(x_test)

metric_result = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"evaluation metric (MAPE %): {metric_result}")

# Save the model
dump(
    best_data_model_pipeline,
    "data/06_models/diamantes_price-hist_gradient_boosting-v1.joblib",
    protocol=5,
)
