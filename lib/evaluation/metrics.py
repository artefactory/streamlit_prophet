import pandas as pd
import numpy as np
from lib.evaluation.preparation import add_time_groupers


def MAPE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Percentage Error (MAPE). Must be multiplied by 100 to get percentage.
    Args:
        y_true (pd.Series): ground truth Y series
        y_pred (pd.Series): prediction Y series
    Returns:
        float: MAPE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    # TODO : Gestion du cas où y_true = 0.
    return mape


def SMAPE(y_test, y_pred):
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


def MSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Square Error (MSE).
    Args:
        y_true (pd.Series): ground truth Y series
        y_pred (pd.Series): prediction Y series
    Returns:
        float: MSE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = ((y_true - y_pred) ** 2).mean()
    return mse


def RMSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root Mean Square Error (RMSE).
    Args:
        y_true (pd.Series): ground truth Y series
        y_pred (pd.Series): prediction Y series
    Returns:
        float: RMSE
    """
    rmse = np.sqrt(MSE(y_true, y_pred))
    return rmse


def MAE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Error (MAE).
    Args:
        y_true (pd.Series): ground truth Y series
        y_pred (pd.Series): prediction Y series
    Returns:
        float: MAE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = abs(y_true - y_pred).mean()
    return mae


def get_perf_metrics(evaluation_df: pd.DataFrame, eval: dict) -> dict:
    perf = dict()
    metrics = {
        'MAPE': MAPE,
        'SMAPE': SMAPE,
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE
    }
    metrics_df = add_time_groupers(evaluation_df)
    if eval['method'] == 'Compute global error':
        metrics_df = metrics_df.groupby(eval['granularity']).agg({'truth': 'sum', 'forecast': 'sum'}).reset_index()
        for m in eval['metrics']:
            metrics_df[m] = metrics_df[['truth', 'forecast']].apply(lambda x: metrics[m](x[0], x[1]), axis=1)
            perf[m] = metrics_df[[eval['granularity'], m]]
    elif eval['method'] == 'Sum all errors':
        # TODO : Implémenter method 'sum all errors'
        import streamlit as st
        st.write('Method not implemented yet.')
    return metrics_df.drop(['truth', 'forecast'], axis=1), perf
