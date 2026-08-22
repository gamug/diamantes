"""Model validation stage — NOT YET IMPLEMENTED.

Reserved for the ``deepchecks`` model-evaluation suite currently run ad hoc
in
``notebooks/5-models/05-gmg-deepcheck-ML-process-2026_08_18.ipynb``
(``deepchecks.tabular.suites.model_evaluation``), which checks things like
train/test performance degradation, weak-segment detection and residual
analysis for a fitted model — complementary to the plain point metrics in
:mod:`model.evaluate`.

Planned shape (to be implemented next):

    def check_model_evaluation(
        model: Any,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> "deepchecks.core.SuiteResult": ...

Should build ``deepchecks.tabular.Dataset`` objects for ``train_df``/
``test_df`` using ``data.constants.CATEGORICAL_ORDINAL_FEATURES`` and
``data.constants.TARGET_COLUMN``, run ``model_evaluation()``, and return (or
raise on) the result so ``pipelines/training_pipeline`` can gate model
promotion on it. See :mod:`data.validation` for the matching data-validation
stub and the deepchecks compatibility shims both will need.
"""
