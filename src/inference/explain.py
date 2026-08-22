"""Per-prediction explanations via SHAP.

The multiplicative decomposition here mirrors the one used in
``notebooks/6-interpretation/01-gmg-modeloX_interpretation-2026_08_18.ipynb``:
because the model is trained on ``log(price)``, each SHAP value ``v``
multiplies the price by ``exp(v)`` rather than adding to it.
"""

from typing import Any

import numpy as np
import pandas as pd
import shap


def explain_prediction(model: Any, diamond: pd.DataFrame) -> pd.DataFrame:
    """Explain a diamond-price prediction as a per-feature price multiplier.

    Args:
        model: A fitted ``TransformedTargetRegressor`` wrapping a
            ``Pipeline`` with ``"preprocessor"`` and ``"model"`` steps (see
            :func:`model.train.build_model_pipeline`); ``"model"`` must be a
            tree-based estimator supported by ``shap.TreeExplainer``.
        diamond: One-row DataFrame with the diamond's characteristics.

    Returns:
        DataFrame with one row per (transformed) feature, sorted by impact,
        with columns ``"feature"``, ``"multiplier"`` (e.g. ``"1.120x"``) and
        ``"effect"`` (``"increases price"``/``"decreases price"``).
    """
    inner_pipeline = model.regressor_
    preprocessor = inner_pipeline.named_steps["preprocessor"]
    regressor = inner_pipeline.named_steps["model"]

    transformed = preprocessor.transform(diamond)
    explainer = shap.TreeExplainer(regressor)
    explanation = explainer(transformed)

    contributions = pd.Series(explanation.values[0], index=preprocessor.get_feature_names_out())
    contributions = contributions.reindex(contributions.abs().sort_values(ascending=False).index)

    return pd.DataFrame(
        {
            "feature": contributions.index,
            "multiplier": [f"{np.exp(v):.3f}x" for v in contributions.to_numpy()],
            "effect": [
                "increases price" if v > 0 else "decreases price" for v in contributions.to_numpy()
            ],
        }
    )


def baseline_price(model: Any, diamond: pd.DataFrame) -> float:
    """Return the SHAP baseline price (the model's expected price with no feature evidence).

    Args:
        model: See :func:`explain_prediction`.
        diamond: One-row DataFrame with the diamond's characteristics; only
            used to build the explainer input, the baseline itself doesn't
            depend on its values.

    Returns:
        The baseline price, in dollars.
    """
    inner_pipeline = model.regressor_
    preprocessor = inner_pipeline.named_steps["preprocessor"]
    regressor = inner_pipeline.named_steps["model"]

    transformed = preprocessor.transform(diamond)
    explainer = shap.TreeExplainer(regressor)
    explanation = explainer(transformed)

    return float(np.exp(explanation.base_values[0]))
