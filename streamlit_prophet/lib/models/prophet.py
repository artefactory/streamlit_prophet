from typing import Any, Dict, Optional, Tuple

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from streamlit_prophet.lib.dataprep.clean import exp_transform
from streamlit_prophet.lib.dataprep.format import check_future_regressors_df
from streamlit_prophet.lib.dataprep.split import make_eval_df, make_future_df
from streamlit_prophet.lib.exposition.preparation import get_df_cv_with_hist
from streamlit_prophet.lib.models.preparation import add_prophet_holidays, get_prophet_cv_horizon
from streamlit_prophet.lib.utils.logging import suppress_stdout_stderr


def instantiate_prophet_model(
    params: Dict[Any, Any], use_regressors: bool = True, dates: Optional[Dict[Any, Any]] = None
) -> Prophet:
    """Instantiates a Prophet model with input parameters.

    Parameters
    ----------
    params : Dict
        Model parameters.
    use_regressors : bool
        Whether or not to add regressors to the model.
    dates : dict
        Dictionary containing all relevant dates for training and forecasting.

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
    if dates:
        model = add_prophet_holidays(model, params["holidays"], dates)

    if use_regressors:
        for regressor in params["regressors"].keys():
            model.add_regressor(
                regressor, prior_scale=params["regressors"][regressor]["prior_scale"]
            )
    return model


def forecast_workflow(
    config: Dict[Any, Any],
    use_cv: bool,
    make_future_forecast: bool,
    evaluate: bool,
    cleaning: Dict[Any, Any],
    resampling: Dict[Any, Any],
    params: Dict[Any, Any],
    dates: Dict[Any, Any],
    datasets: Dict[Any, Any],
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    dimensions: Dict[Any, Any],
    load_options: Dict[Any, Any],
) -> Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
    """Trains a Prophet model and makes a prediction on evaluation data and future data if needed.

    Parameters
    ----------
    config : Dict
        Lib configuration dictionary, containing information about random seed to use for training.
    use_cv : bool
        Whether or not cross-validation is used.
    make_future_forecast : bool
        Whether or not to make a forecast on future dates.
    evaluate : bool
        Whether or not to do a model evaluation.
    cleaning : Dict
        Dataset cleaning specifications.
    resampling : Dict
        Dataset resampling specifications.
    params : Dict
        Model parameters.
    dates : Dict
        Dictionary containing all relevant dates for training and forecasting.
    datasets : Dict
        Dictionary containing all relevant dataframes for training and forecasting.
    df : pd.DataFrame
        Full input dataframe, after cleaning, filtering and resampling.
    date_col : str
        Name of date column.
    target_col : str
        Name of target column.
    dimensions : Dict
        Dictionary containing dimensions information.
    load_options : Dict
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
    models: Dict[Any, Any] = dict()
    forecasts: Dict[Any, Any] = dict()
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
    config: Dict[Any, Any],
    use_cv: bool,
    resampling: Dict[Any, Any],
    params: Dict[Any, Any],
    dates: Dict[Any, Any],
    datasets: Dict[Any, Any],
    models: Dict[Any, Any],
    forecasts: Dict[Any, Any],
) -> Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
    """Trains a Prophet model on training data and makes a prediction on evaluation data.

    Parameters
    ----------
    config : Dict
        Lib configuration dictionary, containing information about random seed to use for training.
    use_cv : bool
        Whether or not cross-validation is used.
    resampling : Dict
        Dataset resampling specifications.
    params : Dict
        Model parameters.
    dates : Dict
        Dictionary containing all relevant dates for training and forecasting.
    datasets : Dict
        Dictionary containing all relevant dataframes for training and forecasting.
    models : Dict
        Dictionary containing instantiated Prophet models.
    forecasts : Dict
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
    config: Dict[Any, Any],
    params: Dict[Any, Any],
    cleaning: Dict[Any, Any],
    dates: Dict[Any, Any],
    datasets: Dict[Any, Any],
    models: Dict[Any, Any],
    forecasts: Dict[Any, Any],
    df: pd.DataFrame,
    resampling: Dict[Any, Any],
    date_col: str,
    target_col: str,
    dimensions: Dict[Any, Any],
    load_options: Dict[Any, Any],
) -> Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
    """Trains a Prophet model on the whole dataset and makes a prediction on future data.

    Parameters
    ----------
    config : Dict
        Lib configuration dictionary, containing information about random seed to use for training.
    params : Dict
        Model parameters.
    cleaning : Dict
        Dataset cleaning specifications.
    dates : Dict
        Dictionary containing all relevant dates for training and forecasting.
    datasets : Dict
        Dictionary containing all relevant dataframes for training and forecasting.
    models : Dict
        Dictionary containing instantiated Prophet models.
    forecasts : Dict
        Dictionary containing the different forecasts.
    df : pd.DataFrame
        Full input dataframe, after cleaning, filtering and resampling.
    resampling : Dict
        Dictionary containing dataset frequency information.
    date_col : str
        Name of date column.
    target_col : str
        Name of target column.
    dimensions : Dict
        Dictionary containing dimensions information.
    load_options : Dict
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
        params,
    )
    models["future"] = instantiate_prophet_model(params, use_regressors=use_regressors, dates=dates)
    models["future"].fit(datasets["full"], seed=config["global"]["seed"])
    forecasts["future"] = models["future"].predict(datasets["future"])
    return datasets, models, forecasts
