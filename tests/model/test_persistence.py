from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from model.persistence import load_model, save_model


def test_save_and_load_model_roundtrip(tmp_path: Path) -> None:
    x_train = np.array([[1.0], [2.0], [3.0]])
    y_train = np.array([10.0, 20.0, 30.0])
    model = LinearRegression().fit(x_train, y_train)

    model_path = tmp_path / "nested" / "model.joblib"
    save_model(model, model_path)
    assert model_path.exists()

    loaded_model = load_model(model_path)
    np.testing.assert_allclose(loaded_model.predict(x_train), model.predict(x_train))
