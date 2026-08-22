"""Rule-based baseline model for diamond price prediction.

Ported from ``notebooks/5-models/01-gmg-base_model-2026_08_18.ipynb``. Any
learned model is expected to clear this baseline by a wide margin before its
added complexity is justified (see ``notebooks/1-data``, Pregunta 7).
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class HeuristicPriceRegressor(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Rule-based diamond pricing model designed from the EDA conclusions.

    ``price ~= carat * base_rate(carat_bin) * clarity_mult * color_mult * cut_mult``

    Every rate/multiplier is learned from the training data in :meth:`fit`;
    nothing is hardcoded. Mirrors, in code, the manual appraisal procedure
    described in ``notebooks/1-data`` (Rapaport-style base rate per carat
    bin, adjusted by grade multipliers).

    Args:
        n_carat_bins: Number of quantile bins to split ``carat`` into when
            estimating the base rate per carat.
    """

    def __init__(self, n_carat_bins: int = 10) -> None:
        self.n_carat_bins = n_carat_bins

    def fit(self, x: pd.DataFrame, y: npt.ArrayLike) -> "HeuristicPriceRegressor":
        """Learn the base rate per carat bin and the grade multipliers.

        Args:
            x: Feature DataFrame; must contain ``carat``, ``clarity``,
                ``color`` and ``cut`` columns.
            y: Target price for each row in ``x``.

        Returns:
            ``self``, fitted.
        """
        x = pd.DataFrame(x).reset_index(drop=True)
        y_series = pd.Series(np.asarray(y), name="price")

        price_per_carat = y_series / x["carat"].to_numpy()

        # carat quantile-bin edges learned from the training carats only
        edges = np.unique(np.quantile(x["carat"], np.linspace(0, 1, self.n_carat_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        self.carat_bin_edges_ = edges

        carat_bin = pd.cut(x["carat"], bins=self.carat_bin_edges_, labels=False, duplicates="drop")
        self.base_rate_ = price_per_carat.groupby(carat_bin).mean()
        self.default_base_rate_ = price_per_carat.mean()

        overall_ppc = price_per_carat.mean()
        self.clarity_mult_ = price_per_carat.groupby(x["clarity"].to_numpy()).mean() / overall_ppc
        self.color_mult_ = price_per_carat.groupby(x["color"].to_numpy()).mean() / overall_ppc
        self.cut_mult_ = price_per_carat.groupby(x["cut"].to_numpy()).mean() / overall_ppc
        return self

    def predict(self, x: pd.DataFrame) -> npt.NDArray[np.floating[Any]]:
        """Predict the price for each row.

        Args:
            x: Feature DataFrame; must contain ``carat``, ``clarity``,
                ``color`` and ``cut`` columns.

        Returns:
            Predicted price per row.
        """
        x = pd.DataFrame(x).reset_index(drop=True)
        carat_bin = pd.cut(x["carat"], bins=self.carat_bin_edges_, labels=False, duplicates="drop")

        base = carat_bin.map(self.base_rate_).astype(float).fillna(self.default_base_rate_)
        clarity_factor = x["clarity"].map(self.clarity_mult_).astype(float).fillna(1.0)
        color_factor = x["color"].map(self.color_mult_).astype(float).fillna(1.0)
        cut_factor = x["cut"].map(self.cut_mult_).astype(float).fillna(1.0)

        result: npt.NDArray[np.floating[Any]] = (
            x["carat"].to_numpy()
            * base.to_numpy()
            * clarity_factor.to_numpy()
            * color_factor.to_numpy()
            * cut_factor.to_numpy()
        )
        return result
