"""Regression evaluation metrics shared across training, baseline and interpretation code.

Consolidates the ``regression_metrics`` helper that was duplicated across
several notebooks (``5-models/01``, ``5-models/04``,
``3-analysis/03-gmg-univariate...``).
"""

import numpy.typing as npt
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)


def regression_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> dict[str, float]:
    """Compute the standard regression metrics used across this project.

    Per ``notebooks/1-data`` (Pregunta 5/6), MAPE is the primary metric —
    the price range spans two orders of magnitude ($326-$18,823), so a
    percentage error is more meaningful than an absolute one — with
    MAE/RMSE/R2 kept as complementary, more familiar metrics.

    Args:
        y_true: Ground-truth prices.
        y_pred: Predicted prices.

    Returns:
        Dict with keys ``"MAE"``, ``"RMSE"``, ``"MAPE_%"`` and ``"R2"``.
    """
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAPE_%": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }
