from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@st.cache(ttl=300)
def remove_empty_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Any]]:
    """Remove columns with strictly less than 2 distinct values in input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe whose columns will be checked and potentially removed.

    Returns
    -------
    pd.DataFrame
        Dataframe with empty columns removed.
    list
        List of columns that have been removed.
    """
    count_cols = df.nunique(dropna=False)
    empty_cols = list(count_cols[count_cols < 2].index)
    return df.drop(empty_cols, axis=1), empty_cols


def print_empty_cols(empty_cols: List[Any]) -> None:
    """Displays a message in streamlit dashboard if the input list is not empty.

    Parameters
    ----------
    empty_cols : list
        List of columns that have been removed.
    """
    L = len(empty_cols)
    if L > 0:
        st.error(
            f'The following column{"s" if L > 1 else ""} ha{"ve" if L > 1 else "s"} been removed because '
            f'{"they have" if L > 1 else "it has"} <= 1 distinct values: {", ".join(empty_cols)}'
        )


@st.cache(suppress_st_warning=True, ttl=300)
def format_date_and_target(
    df_input: pd.DataFrame,
    date_col: str,
    target_col: str,
    config: Dict[Any, Any],
    load_options: Dict[Any, Any],
) -> pd.DataFrame:
    """Formats date and target columns of input dataframe.

    Parameters
    ----------
    df_input : pd.DataFrame
        Input dataframe whose columns will be formatted.
    date_col : str
        Name of date column in input dataframe.
    target_col : str
        Name of target column in input dataframe.
    config : Dict
        Lib configuration dictionary.
    load_options : Dict
        Loading options selected by user.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns formatted.
    """
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    df = _format_date(df, date_col, load_options, config)
    df = _format_target(df, target_col, config)
    df = _rename_cols(df, date_col, target_col)
    return df


def _format_date(
    df: pd.DataFrame, date_col: str, load_options: Dict[Any, Any], config: Dict[Any, Any]
) -> pd.DataFrame:
    """Formats date column of input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe whose columns will be formatted.
    date_col : str
        Name of date column in input dataframe.
    load_options : Dict
        Loading options selected by user.
    config : Dict
        Lib config dictionary containing information about default date format.

    Returns
    -------
    pd.DataFrame
        Dataframe with date column formatted.
    """
    try:
        date_series = pd.to_datetime(df[date_col])
        if __check_date_format(date_series) | (
            config["dataprep"]["date_format"] != load_options["date_format"]
        ):
            date_series = pd.to_datetime(df[date_col], format=load_options["date_format"])
        df[date_col] = date_series
        days_range = (df[date_col].max() - df[date_col].min()).days
        sec_range = (df[date_col].max() - df[date_col].min()).seconds
        if ((days_range < 1) & (sec_range < 1)) | (np.isnan(days_range) & np.isnan(sec_range)):
            st.error(
                "Please select the correct date column (selected column has a time range < 1s)."
            )
            st.stop()
        return df
    except:
        st.error(
            "Please select a valid date format (selected column can't be converted into date)."
        )
        st.stop()


def __check_date_format(date_series: pd.Series) -> bool:
    """Checks whether the date column has been correctly converted to datetime.

    Parameters
    ----------
    date_series : pd.Series
        Date column that has been converted.

    Returns
    -------
    bool
        False if conversion has not worked correctly, True otherwise.
    """
    test1 = date_series.map(lambda x: x.year).nunique() < 2
    test2 = date_series.map(lambda x: x.month).nunique() < 2
    test3 = date_series.map(lambda x: x.day).nunique() < 2
    if test1 & test2 & test3:
        return True
    else:
        return False


def _format_target(df: pd.DataFrame, target_col: str, config: Dict[Any, Any]) -> pd.DataFrame:
    """Formats target column of input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe whose columns will be formatted.
    target_col : str
        Name of target column in input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with date column formatted.
    """
    try:
        df[target_col] = df[target_col].astype("float")
        if df[target_col].nunique() < config["validity"]["min_target_cardinality"]:
            st.error(
                "Please select the correct target column (should be numerical, not categorical)."
            )
            st.stop()
        return df
    except:
        st.error("Please select the correct target column (should be of type int or float).")
        st.stop()


def _rename_cols(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Renames date and target columns of input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe whose columns will be renamed.
    date_col : str
        Name of date column in input dataframe.
    target_col : str
        Name of target column in input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns renamed.
    """
    if (target_col != "y") and ("y" in df.columns):
        df = df.rename(columns={"y": "y_2"})
    if (date_col != "ds") and ("ds" in df.columns):
        df = df.rename(columns={"ds": "ds_2"})
    df = df.rename(columns={date_col: "ds", target_col: "y"})
    return df


# NB: date_col and target_col not used, only added to avoid unexpected caching when their values change
@st.cache(ttl=300)
def filter_and_aggregate_df(
    df_input: pd.DataFrame,
    dimensions: Dict[Any, Any],
    config: Dict[Any, Any],
    date_col: str,
    target_col: str,
) -> Tuple[pd.DataFrame, List[Any]]:
    """Filters and aggregates input dataframe according to dimensions dictionary specifications.

    Parameters
    ----------
    df_input : pd.DataFrame
        Input dataframe that will be filtered and/or aggregated.
    dimensions : Dict
        Filtering and aggregation specifications.
    config : Dict
        Lib configuration dictionary.
    date_col : str
        Name of date column in input dataframe.
    target_col : str
        Name of target column in input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe filtered and/or aggregated.
    list
        List of columns removed from input dataframe.
    """
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    df = _filter(df, dimensions)
    df, cols_to_drop = _format_regressors(df, config)
    df = _aggregate(df, dimensions)
    return df, cols_to_drop


def _filter(df: pd.DataFrame, dimensions: Dict[Any, Any]) -> pd.DataFrame:
    """Filters input dataframe according to dimensions dictionary specifications.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that will be filtered and/or aggregated.
    dimensions : Dict
        Filtering specifications.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    filter_cols = list(set(dimensions.keys()) - {"agg"})
    for col in filter_cols:
        df = df.loc[df[col].isin(dimensions[col])]
    return df.drop(filter_cols, axis=1)


def _format_regressors(df: pd.DataFrame, config: Dict[Any, Any]) -> Tuple[pd.DataFrame, List[Any]]:
    """Format some columns in input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe whose columns will be formatted.
    config : Dict
        Lib configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Formatted dataframe.
    list
        List of columns removed from input dataframe.
    """
    cols_to_drop = []
    for col in set(df.columns) - {"ds", "y"}:
        if df[col].nunique(dropna=False) < 2:
            cols_to_drop.append(col)
        elif df[col].nunique(dropna=False) == 2:
            df[col] = df[col].map(dict(zip(df[col].unique(), [0, 1])))
        elif df[col].nunique() <= config["validity"]["max_cat_reg_cardinality"]:
            df = __one_hot_encoding(df, col)
        else:
            try:
                df[col] = df[col].astype("float")
            except:
                cols_to_drop.append(col)
    return df.drop(cols_to_drop, axis=1), cols_to_drop


def __one_hot_encoding(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Applies one-hot encoding to some columns of input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe whose columns will be one-hot encoded.
    col : list
        List of columns to one-hot encode.

    Returns
    -------
    pd.DataFrame
        One-hot encoded dataframe.
    """
    df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    return df.drop(col, axis=1)


def print_removed_cols(cols_removed: List[Any]) -> None:
    """Displays a message in streamlit dashboard if the input list is not empty.

    Parameters
    ----------
    cols_removed : list
        List of columns that have been removed.
    """
    L = len(cols_removed)
    if L > 0:
        st.error(
            f'The following column{"s" if L > 1 else ""} ha{"ve" if L > 1 else "s"} been removed because '
            f'{"they are" if L > 1 else "it is"} neither the target, '
            f'nor a dimension, nor a potential regressor: {", ".join(cols_removed)}'
        )


def _aggregate(df: pd.DataFrame, dimensions: Dict[Any, Any]) -> pd.DataFrame:
    """Aggregates input dataframe according to dimensions dictionary specifications.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that will be filtered and/or aggregated.
    dimensions : Dict
        Filtering specifications.

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe.
    """
    cols_to_agg = set(df.columns) - {"ds", "y"}
    agg_dict = {col: "mean" if df[col].nunique() > 2 else "max" for col in cols_to_agg}
    agg_dict["y"] = dimensions["agg"].lower()
    return df.groupby("ds").agg(agg_dict).reset_index()


@st.cache(ttl=300)
def format_datetime(df_input: pd.DataFrame, resampling: Dict[Any, Any]) -> pd.DataFrame:
    """Formats date column to datetime in input dataframe.

    Parameters
    ----------
    df_input : pd.DataFrame
        Input dataframe whose date column will be formatted to datetime.
    resampling : Dict
        Dictionary whose "freq" key contains the frequency of input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with date column formatted to datetime.
    """
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    if resampling["freq"][-1] in ["H", "s"]:
        df["ds"] = df["ds"].map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        df["ds"] = pd.to_datetime(df["ds"])
    return df


@st.cache(ttl=300)
def resample_df(df_input: pd.DataFrame, resampling: Dict[Any, Any]) -> pd.DataFrame:
    """Resamples input dataframe according to resampling dictionary specifications.

    Parameters
    ----------
    df_input : pd.DataFrame
        Input dataframe that will be resampled.
    resampling : Dict
        Resampling specifications.

    Returns
    -------
    pd.DataFrame
        Resampled dataframe.
    """
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    if resampling["resample"]:
        cols_to_agg = set(df.columns) - {"ds", "y"}
        agg_dict = {col: "mean" if df[col].nunique() > 2 else "max" for col in cols_to_agg}
        agg_dict["y"] = resampling["agg"].lower()
        df = df.set_index("ds").resample(resampling["freq"][-1]).agg(agg_dict).reset_index()
    return df


def check_dataset_size(df: pd.DataFrame, config: Dict[Any, Any]) -> None:
    """Displays a message in streamlit dashboard and stops it if the input dataframe has not enough rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    config : Dict
        Lib configuration dictionary where the minimum number of rows is given.
    """
    if (
        len(df)
        <= config["validity"]["min_data_points_train"] + config["validity"]["min_data_points_val"]
    ):
        st.error(
            f"The dataset has not enough data points ({len(df)} data points only) to make a forecast. "
            f"Please resample with a higher frequency or change cleaning options."
        )
        st.stop()


def check_future_regressors_df(
    datasets: Dict[Any, Any],
    dates: Dict[Any, Any],
    params: Dict[Any, Any],
    resampling: Dict[Any, Any],
    date_col: str,
    dimensions: Dict[Any, Any],
) -> bool:
    """Displays a message if the future regressors dataframe is incorrect and says whether or not to use it afterwards.

    Parameters
    ----------
    datasets : Dict
        Dictionary storing all dataframes.
    dates : Dict
        Dictionary containing future forecasting dates information.
    params : Dict
        Dictionary containing all model parameters and list of selected regressors.
    resampling : Dict
        Dictionary containing dataset frequency information.
    date_col : str
        Name of date column.
    dimensions : Dict
        Dictionary containing dimensions information.

    Returns
    -------
    bool
        Whether or not to use regressors for future forecast.
    """
    use_regressors = False
    if "future_regressors" in datasets.keys():
        # Check date column
        if date_col not in datasets["future_regressors"].columns:
            st.error(
                f"Date column '{date_col}' not found in the dataset provided for future regressors."
            )
            st.stop()
        # Check number of distinct dates
        N_dates_input = datasets["future_regressors"][date_col].nunique()
        N_dates_expected = len(
            pd.date_range(
                start=dates["forecast_start_date"],
                end=dates["forecast_end_date"],
                freq=resampling["freq"],
            )
        )
        if N_dates_input != N_dates_expected:
            st.error(
                f"The dataset provided for future regressors has the right number of distinct dates "
                f"(expected {N_dates_expected}, found {N_dates_input}). "
                f"Please make sure that the date column goes from {dates['forecast_start_date'].strftime('%Y-%m-%d')} "
                f"to {dates['forecast_end_date'].strftime('%Y-%m-%d')} at frequency {resampling['freq']} "
                f"without skipping any date in this range."
            )
            st.stop()
        # Check regressors
        regressors_expected = set(params["regressors"].keys())
        input_cols = set(datasets["future_regressors"])
        if len(input_cols.intersection(regressors_expected)) != len(regressors_expected):
            missing_regressors = [reg for reg in regressors_expected if reg not in input_cols]
            if len(missing_regressors) > 1:
                st.error(
                    f"Columns {', '.join(missing_regressors[:-1])} and {missing_regressors[-1]} are missing "
                    f"in the dataset provided for future regressors."
                )
            else:
                st.error(
                    f"Column {missing_regressors[0]} is missing in the dataset provided for future regressors."
                )
            st.stop()
        # Check dimensions
        dim_expected = {dim for dim in dimensions.keys() if dim != "agg"}
        if len(input_cols.intersection(dim_expected)) != len(dim_expected):
            missing_dim = [dim for dim in dim_expected if dim not in input_cols]
            if len(missing_dim) > 1:
                st.error(
                    f"Dimension columns {', '.join(missing_dim[:-1])} and {missing_dim[-1]} are missing "
                    f"in the dataset provided for future regressors."
                )
            else:
                st.error(
                    f"Dimension column {missing_dim[0]} is missing in the dataset provided for future regressors."
                )
            st.stop()
        use_regressors = True
    return use_regressors


def prepare_future_df(
    datasets: Dict[Any, Any],
    dates: Dict[Any, Any],
    date_col: str,
    target_col: str,
    dimensions: Dict[Any, Any],
    load_options: Dict[Any, Any],
    config: Dict[Any, Any],
    resampling: Dict[Any, Any],
    params: Dict[Any, Any],
) -> Tuple[pd.DataFrame, Dict[Any, Any]]:
    """Applies data preparation to the dataset provided with future regressors.

    Parameters
    ----------
    datasets : Dict
        Dictionary storing all dataframes.
    dates : Dict
        Dictionary containing future forecasting dates information.
    date_col : str
        Name of date column.
    target_col : str
        Name of target column.
    dimensions : Dict
        Dictionary containing dimensions information.
    load_options : Dict
        Loading options selected by user.
    config : Dict
        Lib configuration dictionary.
    resampling : Dict
        Resampling specifications.
    params : Dict
        Dictionary containing all model parameters

    Returns
    -------
    pd.DataFrame
        Prepared  future dataframe.
    dict
        Dictionary storing all dataframes.
    """
    if "future_regressors" in datasets.keys():
        future = datasets["future_regressors"]
        future[target_col] = 0
        future = pd.concat([datasets["uploaded"][list(future.columns)], future], axis=0)
        future, _ = remove_empty_cols(future)
        future = format_date_and_target(future, date_col, target_col, config, load_options)
        future, _ = filter_and_aggregate_df(future, dimensions, config, date_col, target_col)
        future = format_datetime(future, resampling)
        future = resample_df(future, resampling)
        datasets["full"] = future.loc[future["ds"] < dates["forecast_start_date"]]
        future = future.drop("y", axis=1)
    else:
        future_dates = pd.date_range(
            start=datasets["full"].ds.min(),
            end=dates["forecast_end_date"],
            freq=dates["forecast_freq"],
        )
        future = pd.DataFrame(future_dates, columns=["ds"])
    future = add_cap_and_floor_cols(future, params)
    return future, datasets


@st.cache(ttl=300)
def add_cap_and_floor_cols(df_input: pd.DataFrame, params: Dict[Any, Any]) -> pd.DataFrame:
    """Resamples input dataframe according to resampling dictionary specifications.

    Parameters
    ----------
    df_input : pd.DataFrame
        Input dataframe that will be resampled.
    params : Dict
        Model parameters.

    Returns
    -------
    pd.DataFrame
        Dataframe with cap and floor columns if specified.
    """
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    if params["other"]["growth"] == "logistic":
        df["cap"] = params["saturation"]["cap"]
        df["floor"] = params["saturation"]["floor"]
    return df
