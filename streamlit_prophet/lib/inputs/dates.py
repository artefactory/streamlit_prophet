from typing import Any, Dict, Tuple

import datetime
from datetime import timedelta

import pandas as pd
import streamlit as st
from streamlit_prophet.lib.dataprep.split import (
    get_cv_cutoffs,
    get_max_possible_cv_horizon,
    get_train_end_date_default_value,
    print_cv_folds_dates,
    print_forecast_dates,
    raise_error_cv_dates,
)
from streamlit_prophet.lib.utils.mapping import (
    convert_into_nb_of_days,
    convert_into_nb_of_seconds,
    mapping_freq_names,
)


def input_train_dates(
    df: pd.DataFrame,
    use_cv: bool,
    config: Dict[Any, Any],
    resampling: Dict[Any, Any],
    dates: Dict[Any, Any],
) -> Dict[Any, Any]:
    """Lets the user enter training dates.

    Parameters
    ----------
    df : Dict
        Prepared dataset (after filtering, resampling, cleaning).
    use_cv : bool
        Whether or not cross-validation is used.
    config : Dict
        Lib config dictionary containing information needed to set default dates displayed in streamlit.
    resampling : Dict
        Dictionary containing dataset frequency information.
    dates : Dict
        Empty dictionary.

    Returns
    -------
    dict
        Dictionary containing training dates information.
    """
    col1, col2 = st.columns(2)
    set_name = "CV" if use_cv else "Training"
    dates["train_start_date"] = col1.date_input(
        f"{set_name} start date", value=df.ds.min(), min_value=df.ds.min(), max_value=df.ds.max()
    )
    default_end_date = get_train_end_date_default_value(df, dates, resampling, config, use_cv)
    dates["train_end_date"] = col2.date_input(
        f"{set_name} end date",
        value=default_end_date,
        min_value=dates["train_start_date"] + timedelta(days=1),
        max_value=df.ds.max(),
    )
    return dates


def input_val_dates(
    df: pd.DataFrame, dates: Dict[Any, Any], config: Dict[Any, Any]
) -> Dict[Any, Any]:
    """Lets the user enter validation dates.

    Parameters
    ----------
    df : Dict
        Prepared dataset (after filtering, resampling, cleaning).
    dates : Dict
        Dictionary containing training dates information.
    config : Dict
        Lib config dictionary containing information needed to set default dates displayed in streamlit.

    Returns
    -------
    dict
        Dictionary containing training and validation dates information.
    """
    col1, col2 = st.columns(2)
    dates["val_start_date"] = col1.date_input(
        "Validation start date",
        value=dates["train_end_date"] + timedelta(days=config["split"]["gap_train_valid"]),
        min_value=dates["train_end_date"] + timedelta(days=1),
        max_value=df.ds.max(),
    )
    dates["val_end_date"] = col2.date_input(
        "Validation end date",
        value=df.ds.max(),
        min_value=dates["val_start_date"] + timedelta(days=1),
        max_value=df.ds.max(),
    )
    return dates


def input_cv(
    dates: Dict[Any, Any],
    resampling: Dict[Any, Any],
    config: Dict[Any, Any],
    readme: Dict[Any, Any],
) -> Dict[Any, Any]:
    """Lets the user enter cross-validation specifications.

    Parameters
    ----------
    dates : Dict
        Dictionary containing training dates information.
    resampling : Dict
        Dictionary containing dataset frequency information.
    config : Dict
        Lib config dictionary containing information needed to set default dates displayed in streamlit.
    readme : Dict
        Dictionary containing tooltips to guide user's choices.

    Returns
    -------
    dict
        Dictionary containing training dates and cross-validation specifications.
    """
    dates["n_folds"] = st.number_input(
        "Number of CV folds",
        min_value=1,
        value=config["split"]["CV"],
        help=readme["tooltips"]["cv_n_folds"],
    )
    freq = resampling["freq"][-1]
    max_possible_horizon = get_max_possible_cv_horizon(dates, resampling)
    dates["folds_horizon"] = st.number_input(
        f"Horizon of each fold (in {mapping_freq_names(freq)})",
        min_value=3,
        max_value=max_possible_horizon,
        value=min(config["horizon"][freq], max_possible_horizon),
        help=readme["tooltips"]["cv_horizon"],
    )
    dates["cutoffs"] = get_cv_cutoffs(dates, freq)
    print_cv_folds_dates(dates, freq)
    raise_error_cv_dates(dates, resampling, config)
    return dates


def input_forecast_dates(
    df: pd.DataFrame,
    dates: Dict[Any, Any],
    resampling: Dict[Any, Any],
    config: Dict[Any, Any],
    readme: Dict[Any, Any],
) -> Dict[Any, Any]:
    """Lets the user enter future forecast dates.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset (after filtering, resampling, cleaning).
    dates : Dict
        Dictionary containing dates information.
    resampling : Dict
        Dictionary containing dataset frequency information.
    config : Dict
        Lib config dictionary containing information needed to set default dates displayed in streamlit.
    readme : Dict
        Dictionary containing tooltips to guide user's choices.

    Returns
    -------
    dict
        Dictionary containing future forecast dates information.
    """
    forecast_freq_name = mapping_freq_names(resampling["freq"][-1])
    dates["forecast_horizon"] = st.number_input(
        f"Forecast horizon in {forecast_freq_name}",
        min_value=1,
        value=config["horizon"][resampling["freq"][-1]],
        help=readme["tooltips"]["forecast_horizon"],
    )
    if forecast_freq_name in ["seconds", "hours"]:
        dates["forecast_start_date"] = df.ds.max() + timedelta(seconds=1)
        timedelta_horizon = convert_into_nb_of_seconds(
            resampling["freq"][-1], dates["forecast_horizon"]
        )
        dates["forecast_end_date"] = dates["forecast_start_date"] + timedelta(
            seconds=timedelta_horizon
        )
    else:
        dates["forecast_start_date"] = df.ds.max() + timedelta(days=1)
        timedelta_horizon = convert_into_nb_of_days(
            resampling["freq"][-1], dates["forecast_horizon"]
        )
        dates["forecast_end_date"] = dates["forecast_start_date"] + timedelta(
            days=timedelta_horizon
        )
    dates["forecast_freq"] = str(resampling["freq"])
    print_forecast_dates(dates, resampling)
    return dates


def input_waterfall_dates(
    forecast_df: pd.DataFrame, resampling: Dict[Any, Any]
) -> Tuple[datetime.date, datetime.date]:
    """Lets the user enter dates for waterfall components chart.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Dataframe with forecast and components.
    resampling : Dict
        Resampling specifications (granularity, dataset frequency).

    Returns
    -------
    datetime.date
        Waterfall start date.
    datetime.date
        Waterfall end date.
    """
    max_date = forecast_df.loc[~pd.isnull(forecast_df["trend"])]["ds"].max()
    col1, col2 = st.columns(2)
    start_date = col1.date_input(
        "Start date", value=forecast_df.ds.min(), min_value=forecast_df.ds.min(), max_value=max_date
    )
    freq = resampling["freq"][-1]
    n_periods = col2.number_input(
        f"Number of {mapping_freq_names(freq)} to focus on", value=1, min_value=1
    )
    end_date = start_date + timedelta(days=convert_into_nb_of_days(freq, n_periods))
    return start_date, end_date
