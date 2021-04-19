import pandas as pd
from typing import List, Optional
import numpy as np

def format_columns(
        df: pd.DataFrame,
        date_col: str,
        target_col: str
        ) -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    df[target_col] = df[target_col].astype('float')
    df = df.rename(columns={date_col: 'ds', target_col:'y'})
    return df

def clean_timeseries(
        df: pd.DataFrame,
        del_negative: bool,
        del_zeros: bool,
        del_days: Optional[List[int]],
        ) -> pd.DataFrame:
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
    if del_negative is True:
        df_clean['__to_remove'] = np.where(df_clean['y'] < 0, 1, df_clean['__to_remove'])
    if del_days is not None:
        df_clean['__to_remove'] = np.where(
            df_clean.ds.dt.dayofweek.isin(del_days), 1, df_clean['__to_remove'])
    if del_zeros is True:
        df_clean['__to_remove'] = np.where(df_clean['y'] == 0, 1, df_clean['__to_remove'])
    # then, process the data and delete the flag
    df_clean = df_clean.query("__to_remove != 1")
    del df_clean["__to_remove"]
    return df_clean