import pandas as pd
import streamlit as st


@st.cache(suppress_st_warning=True)
def format_date_and_target(df_input: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    try:
        df = df_input.copy() # To avoid CachedObjectMutationWarning
        df[date_col] = pd.to_datetime(df[date_col])
        df[target_col] = df[target_col].astype('float')
        df = df.rename(columns={date_col: 'ds', target_col: 'y'})
    except:
        st.write('Please select the correct date and target columns.')
        st.stop()
    return df


@st.cache()
def filter_and_aggregate_df(df: pd.DataFrame, dimensions: dict):
    df = _filter(df, dimensions)
    df = _format_regressors(df)
    df = _aggregate(df)
    return df


@st.cache()
def resample_df(df: pd.DataFrame, resampling: dict):
    freq = resampling['freq']
    cols_to_agg = set(df.columns) - set(['ds'])
    agg_dict = {col: 'mean' if df[col].nunique() > 2 else 'max' for col in cols_to_agg}
    agg_dict['y'] = 'sum'
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




