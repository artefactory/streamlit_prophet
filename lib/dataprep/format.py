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


def filter_and_aggregate_df(df: pd.DataFrame, dimensions: dict) -> pd.DataFrame:
    df = _filter(df, dimensions)
    df, cols_to_drop = _format_regressors(df)
    _print_removed_cols(cols_to_drop)
    df = _aggregate(df, dimensions)
    return df


@st.cache()
def _filter(df_input: pd.DataFrame, dimensions: dict) -> pd.DataFrame:
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    filter_cols = list(set(dimensions.keys()) - set(['agg']))
    for col in filter_cols:
        df = df.loc[df[col].isin(dimensions[col])]
    return df.drop(filter_cols, axis=1)


@st.cache()
def _format_regressors(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = []
    for col in set(df.columns) - set(['ds', 'y']):
        if df[col].nunique() < 2:
            cols_to_drop.append(col)
        elif df[col].nunique() == 2: # TODO : One hot encoding si cardinalité > 2 et < à un seuil ?
            df[col] = df[col].map(dict(zip(df[col].unique(), [0, 1])))
        else:
            try:
                df[col] = df[col].astype('float')
            except:
                cols_to_drop.append(col)
    return df.drop(cols_to_drop, axis=1), cols_to_drop


def _print_removed_cols(cols_to_drop):
    L = len(cols_to_drop)
    if L > 0:
        st.error(f'The following column{"s" if L>1 else ""} ha{"ve" if L>1 else "s"} been removed because '
                 f'{"they are" if L>1 else "it is"} neither the target, '
                 f'nor a dimension, nor a potential regressor: {", ".join(cols_to_drop)}')


@st.cache()
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
