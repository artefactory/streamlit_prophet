import pandas as pd
import numpy as np


def format_date_and_target(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    df[target_col] = df[target_col].astype('float')
    df = df.rename(columns={date_col: 'ds', target_col: 'y'})
    return df


def clean_df(df: pd.DataFrame, cleaning: dict) -> pd.DataFrame:
    df = _remove_rows(df, cleaning)
    df = _log_transform(df, cleaning)
    return df


def _log_transform(df: pd.DataFrame, cleaning: dict) -> pd.DataFrame:
    if cleaning['log_transform']:
        df['y'] = np.log(df['y'])
    return df


def _remove_rows(df: pd.DataFrame, cleaning: dict) -> pd.DataFrame:
    """
    Parameters
    ----------
    - df : DataFrame
        DataFrame with Y
    - target_col : str
        name of the column that contains the values
    - del_negative : bool
        if True, will clean negative y values
    - del_zeros : bool
        if True, clean rows where y=0
    - del_days : List[integers], Optional
        Clean specified day(s). 0 for Monday, 6 for Sunday.
    """
    # first, let's flag values that needs to be processed
    df_clean = df.copy()
    df_clean['__to_remove'] = 0
    if cleaning['del_negative']:
        df_clean['__to_remove'] = np.where(df_clean['y'] < 0, 1, df_clean['__to_remove'])
    if cleaning['del_days'] is not None:
        df_clean['__to_remove'] = np.where(
            df_clean.ds.dt.dayofweek.isin(cleaning['del_days']), 1, df_clean['__to_remove'])
    if cleaning['del_zeros']:
        df_clean['__to_remove'] = np.where(df_clean['y'] == 0, 1, df_clean['__to_remove'])
    # then, process the data and delete the flag
    df_clean = df_clean.query("__to_remove != 1")
    del df_clean["__to_remove"]
    return df_clean


def exp_transform(datasets: dict, forecasts:dict):
    for data in set(datasets.keys()):
        if 'y' in datasets[data].columns:
            df_exp = datasets[data].copy()
            df_exp['y'] = np.exp(df_exp['y'])
            datasets[data] = df_exp.copy()
    for data in set(forecasts.keys()):
        # TODO : Gérer le passage à l'exponentiel de trend / seasonalities / regressors
        if 'yhat' in forecasts[data].columns:
            df_exp = forecasts[data].copy()
            df_exp['yhat'] = np.exp(df_exp['yhat'])
            forecasts[data] = df_exp.copy()
    return datasets, forecasts
