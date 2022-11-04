from typing import Any, Dict

import pandas as pd
from prophet import Prophet
from streamlit_prophet.lib.utils.holidays import lockdown_format_func
from streamlit_prophet.lib.utils.mapping import (
    COVID_LOCKDOWN_DATES_MAPPING,
    SCHOOL_HOLIDAYS_FUNC_MAPPING,
    convert_into_nb_of_days,
    convert_into_nb_of_seconds,
)


def get_prophet_cv_horizon(dates: Dict[Any, Any], resampling: Dict[Any, Any]) -> str:
    """Returns cross-validation horizon at the right format for Prophet cross_validation function.

    Parameters
    ----------
    dates : Dict
        Dictionary containing cross-validation information.
    resampling : Dict
        Dictionary containing dataset frequency information.

    Returns
    -------
    str
        Cross-validation horizon at the right format for Prophet cross_validation function.
    """
    freq = resampling["freq"][-1]
    horizon = dates["folds_horizon"]
    if freq in ["s", "H"]:
        prophet_horizon = f"{convert_into_nb_of_seconds(freq, horizon)} seconds"
    else:
        prophet_horizon = f"{convert_into_nb_of_days(freq, horizon)} days"
    return prophet_horizon


def add_prophet_holidays(
    model: Prophet, holidays_params: Dict[Any, Any], dates: Dict[Any, Any]
) -> pd.DataFrame:
    """Add all available holidays to the Prophet model

    Parameters
    ----------
    model: Prophet
        Prophet model to add holidays to
    holidays_params: dict
        dict of parameters including 'country': str, 'public_holidays': bool, 'school_holidays': bool, lockdown_events: List[int]
    dates : dict
        Dictionary containing all relevant dates for training and forecasting.

    Returns
    -------
    Prophet
        Prophet model with holidays added
    """
    country = holidays_params["country"]
    if holidays_params["public_holidays"]:
        model.add_country_holidays(country)

    holidays_df_list = []
    if holidays_params["school_holidays"]:
        all_dates = {
            k: v
            for k, v in dates.items()
            if k not in ["n_folds", "folds_horizon", "forecast_horizon", "cutoffs", "forecast_freq"]
        }
        years = list(range(min(all_dates.values()).year, max(all_dates.values()).year + 1))
        get_holidays_func = SCHOOL_HOLIDAYS_FUNC_MAPPING[country]
        holidays_df = get_holidays_func(years)
        holidays_df[["lower_window", "upper_window"]] = 0
        holidays_df_list.append(holidays_df)

    for lockdown_idx in holidays_params["lockdown_events"]:
        start, end = COVID_LOCKDOWN_DATES_MAPPING[country][lockdown_idx]
        lockdown_df = pd.DataFrame(
            {
                "holiday": lockdown_format_func(lockdown_idx),
                "ds": pd.date_range(start=start, end=end),
                "lower_window": 0,
                "upper_window": 0,
            }
        )
        holidays_df_list.append(lockdown_df)

    if len(holidays_df_list) == 0:
        return model
    holidays_df = pd.concat(holidays_df_list, sort=True)
    model.holidays = holidays_df
    return model
