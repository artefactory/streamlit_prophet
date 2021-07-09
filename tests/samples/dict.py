from typing import Any, Dict, List

from datetime import datetime

import numpy as np
import pandas as pd
from streamlit_prophet.lib.dataprep.split import get_cv_cutoffs
from streamlit_prophet.lib.inputs.dataprep import _autodetect_dimensions
from streamlit_prophet.lib.utils.load import load_config

config, _, _ = load_config(
    "config_streamlit.toml", "config_instructions.toml", "config_readme.toml"
)


# Resampling
def make_resampling_test(
    freq: str = "D", resample: bool = True, agg: str = "Mean"
) -> Dict[Any, Any]:
    """Creates a resampling dictionary with specifications defined by the arguments, for testing purpose.

    Parameters
    ----------
    freq : str
        Value for the 'freq' key of dictionary.
    resample : bool
        Value for the 'resample' key of dictionary.
    agg : str
        Value for the 'agg' key of dictionary.

    Returns
    -------
    dict
        Resampling dictionary that will be used for unit tests.
    """
    return {"freq": freq, "resample": resample, "agg": agg}


# Dimensions
def make_dimensions_test(df: pd.DataFrame, frac: float = 0.5, agg: Any = "Mean") -> Dict[Any, Any]:
    """Creates a dimensions dictionary with specifications defined by the arguments, for testing purpose.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe before filtering step.
    frac : float
        Value for the 'frac' key of dictionary.
    agg : str
        Value for the 'agg' key of dictionary.

    Returns
    -------
    dict
        Dimensions dictionary that will be used for unit tests.
    """
    dimensions = dict()
    dimensions_cols = _autodetect_dimensions(df)
    for col in dimensions_cols:
        values = list(df[col].unique())
        dimensions[col] = np.random.choice(values, size=int(len(values) * frac), replace=False)
    if len(dimensions_cols) > 0:
        dimensions["agg"] = agg
    return dimensions


# Cleaning
def make_cleaning_test(
    del_days: List[Any] = [],
    del_zeros: bool = True,
    del_negative: bool = True,
    log_transform: bool = False,
) -> Dict[Any, Any]:
    """Creates a cleaning dictionary with specifications defined by the arguments, for testing purpose.

    Parameters
    ----------
    del_days : list
        Value for the 'del_days' key of dictionary.
    del_zeros : bool
        Value for the 'del_zeros' key of dictionary.
    del_negative : bool
        Value for the 'del_negative' key of dictionary.
    log_transform : bool
        Value for the 'log_transform' key of dictionary.

    Returns
    -------
    dict
        Cleaning dictionary that will be used for unit tests.
    """
    return {
        "del_days": del_days,
        "del_zeros": del_zeros,
        "del_negative": del_negative,
        "log_transform": log_transform,
    }


# Dates
def make_dates_test(
    train_start: str = "2010-01-01",
    train_end: str = "2014-12-31",
    val_start: str = "2015-01-01",
    val_end: str = "2019-12-31",
    forecast_start: str = "2020-01-01",
    forecast_end: str = "2020-12-31",
    n_folds: int = 5,
    folds_horizon: int = 0,
    freq: str = "D",
) -> Dict[Any, Any]:
    """Creates a dates dictionary with specifications defined by the arguments, for testing purpose.

    Parameters
    ----------
    train_start : str
        Value for the 'train_start_date' key of dictionary.
    train_end : str
        Value for the 'train_end_date' key of dictionary.
    val_start : str
        Value for the 'val_start_date' key of dictionary.
    val_end : str
        Value for the 'val_end_date' key of dictionary.
    forecast_start : str
        Value for the 'forecast_start_date' key of dictionary.
    forecast_end : str
        Value for the 'forecast_end_date' key of dictionary.
    n_folds : int
        Value for the 'n_folds' key of dictionary.
    folds_horizon : int
        Value for the 'folds_horizon' key of dictionary.
    freq : str
        Value for the 'freq' key of dictionary.

    Returns
    -------
    dict
        Dates dictionary that will be used for unit tests.
    """
    dates = {
        "train_start_date": datetime.date(datetime.strptime(train_start, "%Y-%m-%d")),
        "train_end_date": datetime.date(datetime.strptime(train_end, "%Y-%m-%d")),
        "val_start_date": datetime.date(datetime.strptime(val_start, "%Y-%m-%d")),
        "val_end_date": datetime.date(datetime.strptime(val_end, "%Y-%m-%d")),
        "forecast_start_date": datetime.date(datetime.strptime(forecast_start, "%Y-%m-%d")),
        "forecast_end_date": datetime.date(datetime.strptime(forecast_end, "%Y-%m-%d")),
        "forecast_freq": freq,
        "n_folds": n_folds,
        "freq": freq,
        "folds_horizon": config["horizon"][freq[-1]] if folds_horizon == 0 else folds_horizon,
    }
    dates["cutoffs"] = get_cv_cutoffs(dates, freq[-1])
    return dates


# Eval
def make_eval_test(
    granularity: str = "Daily", get_perf_on_agg_forecast: bool = False
) -> Dict[Any, Any]:
    """Creates an evaluation dictionary with specifications defined by the arguments, for testing purpose.

    Parameters
    ----------
    granularity : str
        Value for the 'granularity' key of dictionary.
    get_perf_on_agg_forecast : bool
        Value for the 'get_perf_on_agg_forecast' key of dictionary.

    Returns
    -------
    dict
        Evaluation dictionary that will be used for unit tests.
    """
    return {
        "granularity": granularity,
        "get_perf_on_agg_forecast": get_perf_on_agg_forecast,
        "metrics": ["MAPE", "RMSE", "SMAPE", "MSE", "MAE"],
    }


# Params
def make_params_test(regressors: Dict[Any, Any] = dict()) -> Dict[Any, Any]:
    """Creates a params dictionary with specifications defined by the arguments, for testing purpose.

    Parameters
    ----------
    regressors : Dict
        Value for the 'regressors' key of dictionary.

    Returns
    -------
    dict
        Params dictionary that will be used for unit tests.
    """
    default_params = config["model"]
    return {
        "seasonalities": {"yearly": {"prophet_param": "auto"}, "weekly": {"prophet_param": "auto"}},
        "prior_scale": {
            "changepoint_prior_scale": default_params["changepoint_prior_scale"],
            "seasonality_prior_scale": default_params["seasonality_prior_scale"],
            "holidays_prior_scale": default_params["holidays_prior_scale"],
        },
        "other": {
            "changepoint_range": default_params["changepoint_range"],
            "growth": default_params["growth"][0],
        },
        "holidays": {
            "country": "US",
            "public_holidays": True,
            "school_holidays": False,
            "lockdown_events": [],
        },
        "regressors": regressors,
    }
