import numpy as np
import pandas as pd
import streamlit as st


def clean_df(df: pd.DataFrame, cleaning: dict) -> pd.DataFrame:
    df = _remove_rows(df, cleaning)
    df = _log_transform(df, cleaning)
    return df


def clean_future_df(df: pd.DataFrame, cleaning: dict) -> pd.DataFrame:
    df_clean = df.copy()  # To avoid CachedObjectMutationWarning
    df_clean["__to_remove"] = 0
    if cleaning["del_days"] is not None:
        df_clean["__to_remove"] = np.where(
            df_clean.ds.dt.dayofweek.isin(cleaning["del_days"]), 1, df_clean["__to_remove"]
        )
    df_clean = df_clean.query("__to_remove != 1")
    del df_clean["__to_remove"]
    return df_clean


@st.cache(suppress_st_warning=True)
def _log_transform(df: pd.DataFrame, cleaning: dict) -> pd.DataFrame:
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


@st.cache()
def _remove_rows(df: pd.DataFrame, cleaning: dict) -> pd.DataFrame:
    # first, let's flag values that needs to be processed
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
    # then, process the data and delete the flag
    df_clean = df_clean.query("__to_remove != 1")
    del df_clean["__to_remove"]
    return df_clean


def exp_transform(datasets: dict, forecasts: dict):
    for data in set(datasets.keys()):
        if "y" in datasets[data].columns:
            df_exp = datasets[data].copy()
            df_exp["y"] = np.exp(df_exp["y"])
            datasets[data] = df_exp.copy()
    for data in set(forecasts.keys()):
        # TODO : Gérer le passage à l'exponentiel de trend / seasonalities / regressors
        if "yhat" in forecasts[data].columns:
            df_exp = forecasts[data].copy()
            df_exp["yhat"] = np.exp(df_exp["yhat"])
            forecasts[data] = df_exp.copy()
    return datasets, forecasts
