from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st


def clean_df(df: pd.DataFrame, cleaning: Dict[Any, Any]) -> pd.DataFrame:
    """Cleans the input dataframe according to cleaning dict specifications.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that has to be cleaned.
    cleaning : Dict
        Cleaning specifications.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = _remove_rows(df, cleaning)
    df = _log_transform(df, cleaning)
    return df


def clean_future_df(df: pd.DataFrame, cleaning: Dict[Any, Any]) -> pd.DataFrame:
    """Cleans the input dataframe according to cleaning dict specifications.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that has to be cleaned.
    cleaning : Dict
        Cleaning specifications.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df_clean = df.copy()  # To avoid CachedObjectMutationWarning
    df_clean["__to_remove"] = 0
    if cleaning["del_days"] is not None:
        df_clean["__to_remove"] = np.where(
            df_clean.ds.dt.dayofweek.isin(cleaning["del_days"]), 1, df_clean["__to_remove"]
        )
    df_clean = df_clean.query("__to_remove != 1")
    del df_clean["__to_remove"]
    return df_clean


@st.cache(suppress_st_warning=True, ttl=300)
def _log_transform(df: pd.DataFrame, cleaning: Dict[Any, Any]) -> pd.DataFrame:
    """Applies a log transform to the y column of input dataframe, if possible.
    Raises an error in streamlit dashboard if not possible.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that has to be cleaned.
    cleaning : Dict
        Cleaning specifications.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df_clean = df.copy()  # To avoid CachedObjectMutationWarning
    if cleaning["log_transform"]:
        if df_clean.y.min() <= 0:
            st.error(
                "The target has values <= 0. Please remove negative and 0 values when applying log transform."
            )
            st.stop()
        else:
            df_clean["y"] = np.log(df_clean["y"])
    return df_clean


@st.cache(ttl=300)
def _remove_rows(df: pd.DataFrame, cleaning: Dict[Any, Any]) -> pd.DataFrame:
    """Removes some rows of the input dataframe according to cleaning dict specifications.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that has to be cleaned.
    cleaning : Dict
        Cleaning specifications.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df_clean = df.copy()  # To avoid CachedObjectMutationWarning
    df_clean["__to_remove"] = 0
    if cleaning["del_negative"]:
        df_clean["__to_remove"] = np.where(df_clean["y"] < 0, 1, df_clean["__to_remove"])
    if cleaning["del_days"] is not None:
        df_clean["__to_remove"] = np.where(
            df_clean.ds.dt.dayofweek.isin(cleaning["del_days"]), 1, df_clean["__to_remove"]
        )
    if cleaning["del_zeros"]:
        df_clean["__to_remove"] = np.where(df_clean["y"] == 0, 1, df_clean["__to_remove"])
    df_clean = df_clean.query("__to_remove != 1")
    del df_clean["__to_remove"]
    return df_clean


def exp_transform(
    datasets: Dict[Any, Any], forecasts: Dict[Any, Any]
) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """Applies an exp transform to the y column of dataframes which are values of input dictionaries.

    Parameters
    ----------
    datasets : Dict
        A dictionary whose values are dataframes used as an input to fit a Prophet model.
    forecasts : Dict
        A dictionary whose values are dataframes which are the output of a Prophet prediction.

    Returns
    -------
    dict
        The datasets dictionary with transformed values.
    dict
        The forecasts dictionary with transformed values.
    """
    for data in set(datasets.keys()):
        if "y" in datasets[data].columns:
            df_exp = datasets[data].copy()
            df_exp["y"] = np.exp(df_exp["y"])
            datasets[data] = df_exp.copy()
    for data in set(forecasts.keys()):
        if "yhat" in forecasts[data].columns:
            df_exp = forecasts[data].copy()
            df_exp["yhat"] = np.exp(df_exp["yhat"])
            forecasts[data] = df_exp.copy()
    return datasets, forecasts
