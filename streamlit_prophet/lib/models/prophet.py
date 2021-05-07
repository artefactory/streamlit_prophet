from typing import Tuple

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from streamlit_prophet.lib.dataprep.clean import exp_transform
from streamlit_prophet.lib.dataprep.split import make_eval_df, make_future_df
from streamlit_prophet.lib.exposition.preparation import get_df_cv_with_hist
from streamlit_prophet.lib.utils.logging import suppress_stdout_stderr
from streamlit_prophet.lib.utils.mapping import convert_into_nb_of_days, convert_into_nb_of_seconds


def instantiate_prophet_model(params, use_regressors=True) -> Prophet:
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
    for country in params["holidays"]:
        model.add_country_holidays(country)
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
    cleaning: dict,
    resampling: dict,
    params: dict,
    dates: dict,
    datasets: dict,
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
        datasets, models, forecasts = forecast_eval(
            config, use_cv, resampling, params, dates, datasets, models, forecasts
        )
        if make_future_forecast:
            datasets, models, forecasts = forecast_future(
                config, params, cleaning, dates, datasets, models, forecasts
            )
    if cleaning["log_transform"]:
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
    models["eval"] = instantiate_prophet_model(params)
    models["eval"].fit(datasets["train"], seed=config["global"]["seed"])
    if use_cv:
        forecasts["cv"] = cross_validation(
            models["eval"],
            cutoffs=dates["cutoffs"],
            horizon=_get_prophet_cv_horizon(dates, resampling),
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

    Returns
    -------
    dict
        Dictionary containing all relevant datasets for training and forecasting.
    dict
        Dictionary containing fitted Prophet models.
    dict
        Dictionary containing the different forecasts.
    """
    models["future"] = instantiate_prophet_model(params, use_regressors=False)
    models["future"].fit(datasets["full"], seed=config["global"]["seed"])
    datasets = make_future_df(dates, datasets, cleaning)
    forecasts["future"] = models["future"].predict(datasets["future"])
    return datasets, models, forecasts


def _get_prophet_cv_horizon(dates: dict, resampling: dict) -> str:
    """Returns cross-validation horizon at the right format for Prophet cross_validation function.

    Parameters
    ----------
    dates : dict
        Dictionary containing cross-validation information.
    resampling : dict
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
