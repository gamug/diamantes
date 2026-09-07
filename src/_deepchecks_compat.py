"""Import shims + a stable façade for the bits of ``deepchecks`` this repo uses.

``deepchecks`` 0.19.1 is the final published release (January 2025). It predates
two upstream removals that the pinned scientific stack here has already adopted:

* **numpy >= 2.0** removed the ``np.Inf`` alias (``np.inf`` is the replacement);
  ``deepchecks`` still names ``np.Inf`` while evaluating a class body on import.
* **scikit-learn >= 1.8** dropped the positively-oriented ``"max_error"`` string
  scorer (``"neg_max_error"`` remains); ``deepchecks`` looks ``"max_error"`` up
  at import time.

Neither is fixable from ``deepchecks`` (no newer release exists) and downgrading
numpy / scikit-learn would break the feature and training pipelines. This module
re-adds the two removed names *before* importing ``deepchecks`` and then
re-exports the symbols used elsewhere, so callers can simply do
``from _deepchecks_compat import Dataset, Suite, ...`` and never have to worry
about import order.
"""

from __future__ import annotations

import numpy as np

if not hasattr(np, "Inf"):  # numpy >= 2.0
    np.Inf = np.inf  # type: ignore[attr-defined]

import sklearn.metrics._scorer as _sklearn_scorer
from sklearn.metrics import make_scorer, max_error

if "max_error" not in _sklearn_scorer._SCORERS:  # scikit-learn >= 1.8
    _sklearn_scorer._SCORERS["max_error"] = make_scorer(max_error, greater_is_better=False)

from deepchecks.core import CheckFailure, ConditionCategory, SuiteResult
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (
    DatasetsSizeComparison,
    FeatureDrift,
    FeatureLabelCorrelationChange,
    LabelDrift,
    NewCategoryTrainTest,
    StringMismatchComparison,
    TrainTestSamplesMix,
)

__all__ = [
    "CheckFailure",
    "ConditionCategory",
    "Dataset",
    "DatasetsSizeComparison",
    "FeatureDrift",
    "FeatureLabelCorrelationChange",
    "LabelDrift",
    "NewCategoryTrainTest",
    "StringMismatchComparison",
    "Suite",
    "SuiteResult",
    "TrainTestSamplesMix",
]
