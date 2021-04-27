import pandas as pd
import streamlit as st


@st.cache(suppress_st_warning=True)
def format_date_and_target(df_input: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    df = _format_date(df, date_col)
    df = _format_target(df, target_col)
    df = _rename_cols(df, date_col, target_col)
    return df


def _format_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except:
        st.error('Please select the correct date column.')
        st.stop()
    return df


def _format_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    try:
        df[target_col] = df[target_col].astype('float')
    except:
        st.error('Please select the correct target column (should be of type int or float).')
        st.stop()
    if df[target_col].nunique() < 5:
        st.error('Target column should be numerical, not categorical.')
        st.stop()
    return df


def _rename_cols(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    if (target_col != 'y') and ('y' in df.columns):
        df = df.rename(columns={'y': 'y_2'})
    if (date_col != 'ds') and ('ds' in df.columns):
        df = df.rename(columns={'ds': 'ds_2'})
    df = df.rename(columns={date_col: 'ds', target_col: 'y'})
    return df


@st.cache()
def filter_and_aggregate_df(df: pd.DataFrame, dimensions: dict):
    df = _filter(df, dimensions)
    df = _format_regressors(df)
    df = _aggregate(df)
    return df


@st.cache()
def resample_df(df_input: pd.DataFrame, resampling: dict):
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    freq = resampling['freq']
    if freq[-1] in ['H', 's']:
        df['ds'] = df['ds'].map(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        df['ds'] = pd.to_datetime(df['ds'])
    if resampling['resample']:
        cols_to_agg = set(df.columns) - set(['ds'])
        agg_dict = {col: 'mean' if df[col].nunique() > 2 else 'max' for col in cols_to_agg}
        agg_dict['y'] = 'mean'  # TODO : Variabiliser la fonction d'agrégation ?
        df = df.set_index('ds').resample(freq).agg(agg_dict).reset_index()
    return df


def _filter(df: pd.DataFrame, dimensions: dict):
    for col in dimensions.keys():
        df = df.loc[df[col].isin(dimensions[col])]
    return df.drop(dimensions.keys(), axis=1)


def _aggregate(df: pd.DataFrame):
    cols_to_agg = set(df.columns) - set(['ds'])
    agg_dict = {col: 'mean' if df[col].nunique() > 2 else 'max' for col in cols_to_agg}
    df = df.groupby('ds').agg(agg_dict).reset_index()
    return df


def _format_regressors(df: pd.DataFrame):
    for col in set(df.columns) - set(['ds', 'y']):
        if df[col].nunique() < 2:
            df = df.drop(col, axis=1)
        elif df[col].nunique() == 2:
            df[col] = df[col].map(dict(zip(df[col].unique(), [0, 1])))
        else:
            try:
                df[col] = df[col].astype('float')
            except:
                df = df.drop(col, axis=1)
    return df
