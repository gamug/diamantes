import numpy as np

from model.evaluate import regression_metrics


def test_regression_metrics_perfect_predictions() -> None:
    y_true = [100.0, 200.0, 300.0]
    metrics = regression_metrics(y_true, y_true)

    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["MAPE_%"] == 0.0
    assert metrics["R2"] == 1.0


def test_regression_metrics_known_values() -> None:
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["MAE"] == 15.0
    # MAPE = mean(|10/100|, |20/200|) * 100 = mean(0.10, 0.10) * 100
    assert metrics["MAPE_%"] == 10.0
    assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE_%", "R2"}
