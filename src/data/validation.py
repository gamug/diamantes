"""Data validation stage — NOT YET IMPLEMENTED.

Reserved for the ``deepchecks`` data-integrity and train/test validation
suites currently run ad hoc in
``notebooks/5-models/05-gmg-deepcheck-ML-process-2026_08_18.ipynb``
(``deepchecks.tabular.suites.data_integrity`` and ``train_test_validation``).

Planned shape (to be implemented next):

    def check_data_integrity(df: pd.DataFrame) -> "deepchecks.core.SuiteResult": ...
    def check_train_test_split(train_df: pd.DataFrame, test_df: pd.DataFrame) \
            -> "deepchecks.core.SuiteResult": ...

Both should build a ``deepchecks.tabular.Dataset`` using
``data.constants.CATEGORICAL_ORDINAL_FEATURES`` and
``data.constants.TARGET_COLUMN``, run the corresponding suite, and return
(or raise on) the result so ``pipelines/feature_pipeline`` can gate on it.

Note the numpy 2.0 / scikit-learn >=1.9 compatibility shims that
``deepchecks==0.19.1`` needs at import time (``np.Inf``, ``np.NaN``, the
``max_error`` scorer alias) — see the notebook above for the exact shim code.
"""
