"""Trained-model save/export and load helpers.

Uses joblib (not raw ``pickle``) per scikit-learn's own recommendation for
estimators holding large numpy arrays. Because the fitted pipeline embeds a
``FunctionTransformer(data.features.compute_volume, ...)``
(see :mod:`data.features`), joblib pickles those callables *by reference*
(module + qualified name) rather than by value — so ``data.features`` must be
importable wherever :func:`load_model` is called (training, inference
pipeline, the Streamlit apps). Keeping ``compute_volume`` and
``volume_feature_names_out`` as real module-level functions (as opposed to
notebook-local definitions) is what makes that reference resolvable; see the
project's pre-commit/joblib notes in memory for the failure mode this avoids.
"""

from pathlib import Path
from typing import Any

from joblib import dump, load


def save_model(model: Any, path: Path) -> None:
    """Persist a fitted model/pipeline to disk with joblib.

    Args:
        model: The fitted estimator/pipeline to save.
        path: Destination file, conventionally under ``data/06_models``;
            parent directories are created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path, protocol=5)


def load_model(path: Path) -> Any:
    """Load a model/pipeline previously saved with :func:`save_model`.

    Args:
        path: Path to the ``.joblib`` file.

    Returns:
        The deserialized estimator/pipeline.
    """
    return load(path)
