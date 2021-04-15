from math import sqrt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Percentage Error (MAPE). Must be multiplied by 100 to get percentage.
    Args:
        y_true (pd.Series): ground truth Y series
        y_pred (pd.Series): prediction Y series
    Returns:
        float: MAPE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    return mape


def smape_loss(y_test, y_pred):
    """
    adapted from https://github.com/alan-turing-institute/sktime/blob/15c5ccba8999ddfc52fe37fe4d6a7ff39a19ece3/sktime/performance_metrics/forecasting/_functions.py#L79
    in order to symplify dependancies.
    Symmetric mean absolute percentage error
    Parameters
    ----------
    y_test : pandas Series of shape = (fh,) where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : pandas Series of shape = (fh,)
        Estimated target values.
    Returns
    -------
    loss : float
        sMAPE loss
    """
    nominator = np.abs(y_test - y_pred)
    denominator = np.abs(y_test) + np.abs(y_pred)
    return np.mean(2.0 * nominator / denominator)


def get_perf_metrics(y_true: pd.Series, y_pred: pd.Series, metrics: list) -> dict:
    """Get common regression evaluation metrics
    Args:
        y_test (pd.Series): truth values
        y_pred (pd.Series): prediction values
    Returns:
        dict: dict with common regression evaluation metrics
    """
    all_metrics = {
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'SMAPE': smape_loss(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': sqrt(mean_squared_error(y_true, y_pred)),   # to insure sklearn < 0.22.0 compatibility
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
    }
    return dict((k, all_metrics[k]) for k in metrics)


def prettify_metrics(metrics: dict) -> str:
    output = []
    for name, metric in metrics.items():
        output.append((f'- {name}: {round(metric, 2)}'))
    return '\n'.join(output)