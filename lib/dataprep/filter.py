import pandas as pd


def format_df(df: pd.DataFrame, dimensions: dict):
    df = _filter(df, dimensions)
    df = _format_regressors(df)
    df = _aggregate(df)
    return df


def _filter(df: pd.DataFrame, dimensions: dict):
    for col in dimensions.keys():
        df = df.loc[df[col].isin(dimensions[col])]
    return df.drop(dimensions.keys(), axis=1)


def _aggregate(df: pd.DataFrame):
    cols_to_agg = set(df.columns) - set(['ds'])
    agg_dict = {col: 'sum' if df[col].nunique() > 2 else 'max' for col in cols_to_agg}
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




