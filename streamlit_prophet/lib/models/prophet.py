from typing import Tuple

import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from streamlit_prophet.lib.dataprep.clean import exp_transform
from streamlit_prophet.lib.dataprep.format import check_future_regressors_df
from streamlit_prophet.lib.dataprep.split import make_eval_df, make_future_df
from streamlit_prophet.lib.exposition.preparation import get_df_cv_with_hist
from streamlit_prophet.lib.models.preparation import get_prophet_cv_horizon
from streamlit_prophet.lib.utils.holidays import lockdown_format_func
from streamlit_prophet.lib.utils.logging import suppress_stdout_stderr
from streamlit_prophet.lib.utils.mapping import (
    COVID_LOCKDOWN_DATES_MAPPING,
    SCHOOL_HOLIDAYS_FUNC_MAPPING,
)


def instantiate_prophet_model(params, use_regressors=True, dates=None) -> Prophet:
    """Instantiates a Prophet model with input parameters.

    Parameters
    ----------
    params : dict
        Model parameters.
    use_regressors : bool
        Whether or not to add regressors to the model.

    Returns
    -------
    Prophet
        Instantiated (not fitted) Prophet model.
    """
    seasonality_params = {
        f"{k}_seasonality": params["seasonalities"][k]["prophet_param"]
        for k in {"yearly", "weekly", "daily"}.intersection(set(params["seasonalities"].keys()))
    }
    model = Prophet(**{**params["prior_scale"], **seasonality_params, **params["other"]})
    for _, values in params["seasonalities"].items():
        if "custom_param" in values:
            model.add_seasonality(**values["custom_param"])
    model = _add_prophet_holidays(model, params["holidays"], dates)

    if use_regressors:
        for regressor in params["regressors"].keys():
            model.add_regressor(
                regressor, prior_scale=params["regressors"][regressor]["prior_scale"]
            )
    return model


def forecast_workflow(
    config: dict,
    use_cv: bool,
    make_future_forecast: bool,
    evaluate: bool,
    cleaning: dict,
    resampling: dict,
    params: dict,
    dates: dict,
    datasets: dict,
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    dimensions: dict,
    load_options: dict,
) -> Tuple[dict, dict, dict]:
    """Trains a Prophet model and makes a prediction on evaluation data and future data if needed.

    Parameters
    ----------
    config : dict
        Lib configuration dictionary, containing information about random seed to use for training.
    use_cv : bool
        Whether or not cross-validation is used.
    make_future_forecast : bool
        Whether or not to make a forecast on future dates.
    evaluate : bool
        Whether or not to do a model evaluation.
    cleaning : dict
        Dataset cleaning specifications.
    resampling : dict
        Dataset resampling specifications.
    params : dict
        Model parameters.
    dates : dict
        Dictionary containing all relevant dates for training and forecasting.
    datasets : dict
        Dictionary containing all relevant dataframes for training and forecasting.
    df : pd.DataFrame
        Full input dataframe, after cleaning, filtering and resampling.
    date_col : str
        Name of date column.
    target_col : str
        Name of target column.
    dimensions : dict
        Dictionary containing dimensions information.
    load_options : dict
        Loading options selected by user.

    Returns
    -------
    dict
        Dictionary containing all relevant dataframes for training and forecasting.
    dict
        Dictionary containing fitted Prophet models.
    dict
        Dictionary containing the different forecasts.
    """
    models, forecasts = dict(), dict()
    with suppress_stdout_stderr():
        if evaluate:
            datasets, models, forecasts = forecast_eval(
                config, use_cv, resampling, params, dates, datasets, models, forecasts
            )
        if make_future_forecast:
            datasets, models, forecasts = forecast_future(
                config,
                params,
                cleaning,
                dates,
                datasets,
                models,
                forecasts,
                df,
                resampling,
                date_col,
                target_col,
                dimensions,
                load_options,
            )
    if cleaning["log_transform"] & (evaluate | make_future_forecast):
        datasets, forecasts = exp_transform(datasets, forecasts)
    return datasets, models, forecasts


def forecast_eval(
    config: dict,
    use_cv: bool,
    resampling: dict,
    params: dict,
    dates: dict,
    datasets: dict,
    models: dict,
    forecasts: dict,
) -> Tuple[dict, dict, dict]:
    """Trains a Prophet model on training data and makes a prediction on evaluation data.

    Parameters
    ----------
    config : dict
        Lib configuration dictionary, containing information about random seed to use for training.
    use_cv : bool
        Whether or not cross-validation is used.
    resampling : dict
        Dataset resampling specifications.
    params : dict
        Model parameters.
    dates : dict
        Dictionary containing all relevant dates for training and forecasting.
    datasets : dict
        Dictionary containing all relevant dataframes for training and forecasting.
    models : dict
        Dictionary containing instantiated Prophet models.
    forecasts : dict
        Dictionary containing the different forecasts.

    Returns
    -------
    dict
        Dictionary containing all relevant datasets for training and forecasting.
    dict
        Dictionary containing fitted Prophet models.
    dict
        Dictionary containing the different forecasts.
    """
    models["eval"] = instantiate_prophet_model(params, dates=dates)
    models["eval"].fit(datasets["train"], seed=config["global"]["seed"])
    if use_cv:
        forecasts["cv"] = cross_validation(
            models["eval"],
            cutoffs=dates["cutoffs"],
            horizon=get_prophet_cv_horizon(dates, resampling),
            parallel="processes",
        )
        forecasts["cv_with_hist"] = get_df_cv_with_hist(forecasts, datasets, models)
    else:
        datasets = make_eval_df(datasets)
        forecasts["eval"] = models["eval"].predict(datasets["eval"])
    return datasets, models, forecasts


def forecast_future(
    config: dict,
    params: dict,
    cleaning: dict,
    dates: dict,
    datasets: dict,
    models: dict,
    forecasts: dict,
    df: pd.DataFrame,
    resampling: dict,
    date_col: str,
    target_col: str,
    dimensions: dict,
    load_options: dict,
) -> Tuple[dict, dict, dict]:
    """Trains a Prophet model on the whole dataset and makes a prediction on future data.

    Parameters
    ----------
    config : dict
        Lib configuration dictionary, containing information about random seed to use for training.
    params : dict
        Model parameters.
    cleaning : dict
        Dataset cleaning specifications.
    dates : dict
        Dictionary containing all relevant dates for training and forecasting.
    datasets : dict
        Dictionary containing all relevant dataframes for training and forecasting.
    models : dict
        Dictionary containing instantiated Prophet models.
    forecasts : dict
        Dictionary containing the different forecasts.
    df : pd.DataFrame
        Full input dataframe, after cleaning, filtering and resampling.
    resampling : dict
        Dictionary containing dataset frequency information.
    date_col : str
        Name of date column.
    target_col : str
        Name of target column.
    dimensions : dict
        Dictionary containing dimensions information.
    load_options : dict
        Loading options selected by user.

    Returns
    -------
    dict
        Dictionary containing all relevant datasets for training and forecasting.
    dict
        Dictionary containing fitted Prophet models.
    dict
        Dictionary containing the different forecasts.
    """
    use_regressors = check_future_regressors_df(
        datasets, dates, params, resampling, date_col, dimensions
    )
    datasets = make_future_df(
        dates,
        df,
        datasets,
        cleaning,
        date_col,
        target_col,
        dimensions,
        load_options,
        config,
        resampling,
    )
    models["future"] = instantiate_prophet_model(params, use_regressors=use_regressors)
    models["future"].fit(datasets["full"], seed=config["global"]["seed"])
    forecasts["future"] = models["future"].predict(datasets["future"])
    return datasets, models, forecasts


def _add_prophet_holidays(model: Prophet, holidays_params: dict, dates: dict) -> pd.DataFrame:
    """Add all available holidays fto the Prophet model

    Parameters
    ----------
    model: Prophet
        Prophet model to add holidays to
    holidays_params: dict
        dict of parameters including 'country': str, 'public_holidays': bool, 'school_holidays': bool, lockdown_events: List[int]
    """
    holidays_country = holidays_params["country"]
    if holidays_params["public_holidays"]:
        model.add_country_holidays(holidays_country)

    holidays_df_list = []
    if holidays_params["school_holidays"]:
        years = list(range(min(dates.values()).year, max(dates.values()).year + 1))
        get_holidays_func = SCHOOL_HOLIDAYS_FUNC_MAPPING[holidays_country]
        holidays_df = get_holidays_func(years)
        holidays_df[["lower_window", "upper_window"]] = 0
        holidays_df_list.append(holidays_df)

    for lockdown_idx in holidays_params["lockdown_events"]:
        start, end = COVID_LOCKDOWN_DATES_MAPPING[holidays_country][lockdown_idx]
        ds = pd.date_range(start=start, end=end)
        lockdown_df = pd.DataFrame(
            {
                "holiday": lockdown_format_func(lockdown_idx),
                "ds": ds,
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
