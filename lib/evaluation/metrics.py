import pandas as pd
import numpy as np
from datetime import timedelta
from lib.evaluation.preparation import add_time_groupers
from lib.utils.mapping import convert_into_nb_of_days, convert_into_nb_of_seconds


def MAPE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Percentage Error (MAPE). Must be multiplied by 100 to get percentage.
    Args:
        y_true (pd.Series): ground truth Y series
        y_pred (pd.Series): prediction Y series
    Returns:
        float: MAPE
    """
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = np.where(y_true != 0)[0]
        y_true, y_pred = y_true[mask], y_pred[mask]
        mape = np.mean(np.abs((y_true - y_pred) / y_true))
        return round(mape, 3)
    except:
        return 0


def SMAPE(y_true, y_pred):
    """
    Symmetric mean absolute percentage error
    Args:
        y_true (pd.Series): ground truth Y series
        y_pred (pd.Series): prediction Y series
    Returns:
        float: SMAPE
    """
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = np.where(abs(y_true) + abs(y_pred) != 0)[0]
        y_true, y_pred = y_true[mask], y_pred[mask]
        nominator = np.abs(y_true - y_pred)
        denominator = np.abs(y_true) + np.abs(y_pred)
        smape = np.mean(2.0 * nominator / denominator)
        return round(smape, 3)
    except:
        return 0


def MSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Square Error (MSE).
    Args:
        y_true (pd.Series): ground truth Y series
        y_pred (pd.Series): prediction Y series
    Returns:
        float: MSE
    """
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    mse = ((y_true - y_pred) ** 2).mean()
    return round(mse)


def RMSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root Mean Square Error (RMSE).
    Args:
        y_true (pd.Series): ground truth Y series
        y_pred (pd.Series): prediction Y series
    Returns:
        float: RMSE
    """
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    rmse = np.sqrt(MSE(y_true, y_pred))
    return round(rmse, 1)


def MAE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Error (MAE).
    Args:
        y_true (pd.Series): ground truth Y series
        y_pred (pd.Series): prediction Y series
    Returns:
        float: MAE
    """
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    mae = abs(y_true - y_pred).mean()
    return round(mae, 1)


def get_perf_metrics(evaluation_df: pd.DataFrame, eval: dict, dates: dict, resampling: dict, use_cv: bool):
    metrics = {'MAPE': MAPE, 'SMAPE': SMAPE, 'MSE': MSE, 'RMSE': RMSE, 'MAE': MAE}
    df = _preprocess_eval_df(evaluation_df, use_cv)
    metrics_df = _compute_metrics(df, metrics, eval)
    metrics_df, metrics_dict = _format_eval_results(metrics_df, dates, eval, resampling, use_cv)
    return metrics_df, metrics_dict


def _preprocess_eval_df(evaluation_df: pd.DataFrame, use_cv: bool) -> pd.DataFrame:
    if use_cv:
        df = evaluation_df.copy()
    else:
        df = add_time_groupers(evaluation_df)
    return df


def _compute_metrics(df: pd.DataFrame, metrics: dict, eval: dict) -> pd.DataFrame:
    if eval['get_perf_on_agg_forecast']:
        metrics_df = df.groupby(eval['granularity']).agg({'truth': 'sum', 'forecast': 'sum'}).reset_index()
        for m in eval['metrics']:
            metrics_df[m] = metrics_df[['truth', 'forecast']].apply(lambda x: metrics[m](x.truth, x.forecast), axis=1)
    else:
        metrics_df = pd.DataFrame({eval['granularity']: sorted(df[eval['granularity']].unique())})
        for m in eval['metrics']:
            metrics_df[m] = df.groupby(eval['granularity'])[['truth', 'forecast']]\
                              .apply(lambda x: metrics[m](x.truth, x.forecast))\
                              .sort_index().to_list()
    return metrics_df


def _format_eval_results(metrics_df: pd.DataFrame, dates: dict, eval: dict, resampling: dict, use_cv: bool):
    if use_cv:
        metrics_df = __format_metrics_df_cv(metrics_df, dates, eval, resampling)
        metrics_dict = {m: metrics_df[[eval['granularity'], m]] for m in eval['metrics']}
        metrics_df = __add_avg_std_metrics(metrics_df, eval)
    else:
        metrics_dict = {m: metrics_df[[eval['granularity'], m]] for m in eval['metrics']}
        metrics_df = metrics_df[[eval['granularity']] + eval['metrics']].set_index([eval['granularity']])
    metrics_df = __format_metrics_values(metrics_df, eval)
    return metrics_df, metrics_dict


def __format_metrics_values(metrics_df: pd.DataFrame, eval: dict) -> pd.DataFrame:
    mapping = {'MAPE': '{:,.3f}', 'SMAPE': '{:,.3f}', 'MSE': '{:,.0f}', 'RMSE': '{:,.1f}', 'MAE': '{:,.1f}'}
    for col in eval['metrics']:
        metrics_df[col] = metrics_df[col].map(mapping[col].format)
    return metrics_df


def __format_metrics_df_cv(metrics_df: pd.DataFrame, dates: dict, eval: dict, resampling: dict) -> pd.DataFrame:
    metrics_df = metrics_df.rename(columns={'cutoff': 'Valid Start'})
    freq = resampling['freq'][-1]
    horizon = dates['folds_horizon']
    if freq in ['s', 'H']:
        multiplier = convert_into_nb_of_seconds(freq, 1)
        metrics_df['Valid End'] = metrics_df['Valid Start'].map(lambda x: x + timedelta(seconds=horizon * multiplier))\
                                                           .astype(str)
    else:
        multiplier = convert_into_nb_of_days(freq, 1)
        metrics_df['Valid End'] = metrics_df['Valid Start'].map(lambda x: x + timedelta(days=horizon*multiplier))\
                                                           .astype(str)
    metrics_df['Valid Start'] = metrics_df['Valid Start'].astype(str)
    metrics_df = metrics_df.sort_values('Valid Start', ascending=False).reset_index(drop=True)
    metrics_df[eval['granularity']] = [f"Fold {i}" for i in range(1, len(metrics_df)+1)]
    return metrics_df


def __add_avg_std_metrics(metrics_df: pd.DataFrame, eval: dict) -> pd.DataFrame:
    cols_index = [eval['granularity'], 'Valid Start', 'Valid End']
    metrics_df = metrics_df[cols_index + eval['metrics']].set_index(cols_index)
    metrics_df.loc[('Avg', '', 'Average')] = metrics_df.mean(axis=0)
    metrics_df.loc[('Std', '', '+/-')] = metrics_df.std(axis=0)
    metrics_df = metrics_df.reset_index().set_index(eval['granularity'])
    return metrics_df