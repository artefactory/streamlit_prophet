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


def SMAPE(y_true, y_pred):
    """
    Symmetric mean absolute percentage error
    Args:
        y_true (pd.Series): ground truth Y series
        y_pred (pd.Series): prediction Y series
    Returns:
        float: SMAPE
    """
    nominator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    # TODO : Gestion du cas où y_true = 0 & y_pred = 0.
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
    metrics = {
        'MAPE': MAPE,
        'SMAPE': SMAPE,
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE
    }
    df = add_time_groupers(evaluation_df)
    if eval['get_perf_on_agg_forecast']:
        metrics_df = df.groupby(eval['granularity']).agg({'truth': 'sum', 'forecast': 'sum'}).reset_index()
        for m in eval['metrics']:
            metrics_df[m] = metrics_df[['truth', 'forecast']].apply(lambda x: metrics[m](x[0], x[1]), axis=1)
    else:
        metrics_df = pd.DataFrame({eval['granularity']: sorted(df[eval['granularity']].unique())})
        for m in eval['metrics']:
            metrics_df[m] = df.groupby(eval['granularity'])[['truth', 'forecast']]\
                              .apply(lambda x: metrics[m](x.truth, x.forecast))\
                              .sort_index().to_list()
    perf = {m: metrics_df[[eval['granularity'], m]] for m in eval['metrics']}
    metrics_df = metrics_df[[eval['granularity']] + eval['metrics']]
    return metrics_df, perf
