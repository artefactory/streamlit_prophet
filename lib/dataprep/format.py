import pandas as pd
import numpy as np
import streamlit as st


@st.cache()
def remove_empty_cols(df: pd.DataFrame) -> pd.DataFrame:
    count_cols = df.nunique(dropna=False)
    empty_cols = list(count_cols[count_cols < 2].index)
    return df.drop(empty_cols, axis=1), empty_cols


def print_empty_cols(empty_cols: list):
    L = len(empty_cols)
    if L > 0:
        st.error(f'The following column{"s" if L > 1 else ""} ha{"ve" if L > 1 else "s"} been removed because '
                 f'{"they have" if L > 1 else "it has"} <= 1 distinct values: {", ".join(empty_cols)}')


@st.cache(suppress_st_warning=True)
def format_date_and_target(df_input: pd.DataFrame, date_col: str, target_col: str, config: dict) -> pd.DataFrame:
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    df = _format_date(df, date_col)
    df = _format_target(df, target_col, config)
    df = _rename_cols(df, date_col, target_col)
    return df


def _format_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        days_range = (df[date_col].max() - df[date_col].min()).days
        sec_range = (df[date_col].max() - df[date_col].min()).seconds
        if ((days_range < 1) & (sec_range < 1)) | (np.isnan(days_range) & np.isnan(sec_range)):
            st.error('Please select the correct date column (selected column has a time range < 1s).')
            st.stop()
        return df
    except:
        st.error("Please select the correct date column (selected column can't be converted into date).")
        st.stop()


def _format_target(df: pd.DataFrame, target_col: str, config: dict) -> pd.DataFrame:
    try:
        df[target_col] = df[target_col].astype('float')
        if df[target_col].nunique() < config['validity']['min_target_cardinality']:
            st.error('Please select the correct target column (should be numerical, not categorical).')
            st.stop()
        return df
    except:
        st.error('Please select the correct target column (should be of type int or float).')
        st.stop()


def _rename_cols(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    if (target_col != 'y') and ('y' in df.columns):
        df = df.rename(columns={'y': 'y_2'})
    if (date_col != 'ds') and ('ds' in df.columns):
        df = df.rename(columns={'ds': 'ds_2'})
    df = df.rename(columns={date_col: 'ds', target_col: 'y'})
    return df


# NB: date_col and target_col not used, only added to avoid unexpected caching when their values change
@st.cache()
def filter_and_aggregate_df(df_input: pd.DataFrame, dimensions: dict, config: dict,
                            date_col: str, target_col: str) -> pd.DataFrame:
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    df = _filter(df, dimensions)
    df, cols_to_drop = _format_regressors(df, config)
    df = _aggregate(df, dimensions)
    return df, cols_to_drop


def _filter(df: pd.DataFrame, dimensions: dict) -> pd.DataFrame:
    filter_cols = list(set(dimensions.keys()) - set(['agg']))
    for col in filter_cols:
        df = df.loc[df[col].isin(dimensions[col])]
    return df.drop(filter_cols, axis=1)


def _format_regressors(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cols_to_drop = []
    for col in set(df.columns) - set(['ds', 'y']):
        if df[col].nunique(dropna=False) < 2:
            cols_to_drop.append(col)
        elif df[col].nunique(dropna=False) == 2:
            df[col] = df[col].map(dict(zip(df[col].unique(), [0, 1])))
        elif df[col].nunique() <= config['validity']['max_cat_reg_cardinality']:
            df = __one_hot_encoding(df, col)
        else:
            try:
                df[col] = df[col].astype('float')
            except:
                cols_to_drop.append(col)
    return df.drop(cols_to_drop, axis=1), cols_to_drop


def __one_hot_encoding(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    return df.drop(col, axis=1)


def print_removed_cols(cols_to_drop: list):
    L = len(cols_to_drop)
    if L > 0:
        st.error(f'The following column{"s" if L > 1 else ""} ha{"ve" if L > 1 else "s"} been removed because '
                 f'{"they are" if L > 1 else "it is"} neither the target, '
                 f'nor a dimension, nor a potential regressor: {", ".join(cols_to_drop)}')


def _aggregate(df: pd.DataFrame, dimensions: dict) -> pd.DataFrame:
    cols_to_agg = set(df.columns) - set(['ds', 'y'])
    agg_dict = {col: 'mean' if df[col].nunique() > 2 else 'max' for col in cols_to_agg}
    agg_dict['y'] = dimensions['agg'].lower()
    return df.groupby('ds').agg(agg_dict).reset_index()


@st.cache()
def format_datetime(df_input: pd.DataFrame, resampling: dict) -> pd.DataFrame:
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    if resampling['freq'][-1] in ['H', 's']:
        df['ds'] = df['ds'].map(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        df['ds'] = pd.to_datetime(df['ds'])
    return df


@st.cache()
def resample_df(df_input: pd.DataFrame, resampling: dict) -> pd.DataFrame:
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    if resampling['resample']:
        cols_to_agg = set(df.columns) - set(['ds', 'y'])
        agg_dict = {col: 'mean' if df[col].nunique() > 2 else 'max' for col in cols_to_agg}
        agg_dict['y'] = resampling['agg'].lower()
        df = df.set_index('ds').resample(resampling['freq'][-1]).agg(agg_dict).reset_index()
    return df


def check_dataset_size(df: pd.DataFrame, config: dict):
    if len(df) <= config['split']['min_data_points_train'] + config['split']['min_data_points_val']:
        st.error(f'The dataset has not enough data points ({len(df)} data points only) to make a forecast. '
                 f'Please resample with a higher frequency or change cleaning options.')
        st.stop()
