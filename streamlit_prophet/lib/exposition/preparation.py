from typing import Tuple

from datetime import timedelta

import pandas as pd
from streamlit_prophet.lib.utils.mapping import convert_into_nb_of_days, convert_into_nb_of_seconds


def get_forecast_components(model, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe with only the relevant components to sum to get the prediction.

    Parameters
    ----------
    model : Prophet
        Fitted model.
    forecast_df : pd.DataFrame
        Forecast dataframe returned by Prophet model when predicting on evaluation dataset.

    Returns
    -------
    pd.DataFrame
        Dataframe with only the relevant components to sum to get the prediction.
    """
    fcst = forecast_df.copy()
    components_col_names = get_forecast_components_col_names(fcst) + ["ds"]
    components = fcst[components_col_names]
    for col in components_col_names:
        if col in model.component_modes["multiplicative"]:
            components[col] *= components["trend"]
    components = components.set_index("ds")
    return components


def get_forecast_components_col_names(forecast_df: pd.DataFrame) -> list:
    """Returns the list of columns to keep in forecast dataframe to get all components without upper/lower bounds.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Forecast dataframe returned by Prophet model when predicting on evaluation dataset.

    Returns
    -------
    list
        List of columns to keep in forecast dataframe to get all components without upper/lower bounds.
    """
    components_col = [
        col.replace("_lower", "")
        for col in forecast_df.columns
        if "lower" in col
        and "yhat" not in col
        and "multiplicative" not in col
        and "additive" not in col
    ]
    return components_col


def get_df_cv_with_hist(forecasts: dict, datasets: dict, models: dict) -> pd.DataFrame:
    """Adds training rows not included in CV validation folds to the dataframe containing cross-validation results.

    Parameters
    ----------
    forecasts : dict
        Dictionary containing the dataframe with cross-validation results.
    datasets : dict
        Dictionary containing training dataframe.
    models : dict
        Dictionary containing the model fitted for evaluation.

    Returns
    -------
    pd.DataFrame
        Dataframe containing CV results and predictions on training data not included in CV validation folds.
    """
    df_cv = forecasts["cv"].drop(["cutoff"], axis=1)
    df_past = models["eval"].predict(
        datasets["train"].loc[datasets["train"]["ds"] < df_cv.ds.min()].drop("y", axis=1)
    )
    common_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    df_past = df_past[common_cols + list(set(df_past.columns) - set(common_cols))]
    df_cv = pd.concat([df_cv, df_past], axis=0).sort_values("ds").reset_index(drop=True)
    return df_cv


def get_cv_dates_dict(dates: dict, resampling: dict) -> dict:
    """Returns a dictionary whose keys are CV folds and values are dictionaries with each fold's train/valid dates.

    Parameters
    ----------
    dates : dict
        Dictionary containing cross-validation dates information.
    resampling : dict
        Dictionary containing dataset frequency information.

    Returns
    -------
    dict
        Dictionary containing training and validation dates of each cross-validation fold.
    """
    freq = resampling["freq"][-1]
    train_start = dates["train_start_date"]
    horizon = dates["folds_horizon"]
    cv_dates = dict()
    for i, cutoff in sorted(enumerate(dates["cutoffs"]), reverse=True):
        cv_dates[f"Fold {i + 1}"] = dict()
        cv_dates[f"Fold {i + 1}"]["train_start"] = train_start
        cv_dates[f"Fold {i + 1}"]["val_start"] = cutoff
        cv_dates[f"Fold {i + 1}"]["train_end"] = cutoff
        if freq in ["s", "H"]:
            cv_dates[f"Fold {i + 1}"]["val_end"] = cutoff + timedelta(
                seconds=convert_into_nb_of_seconds(freq, horizon)
            )
        else:
            cv_dates[f"Fold {i + 1}"]["val_end"] = cutoff + timedelta(
                days=convert_into_nb_of_days(freq, horizon)
            )
    return cv_dates


def get_hover_template_cv(cv_dates: dict, resampling: dict) -> Tuple[pd.DataFrame, str]:
    """Returns a dataframe and a dictionary that will be used to show CV folds on a plotly bar plot.

    Parameters
    ----------
    cv_dates : dict
        Dictionary containing training and validation dates of each cross-validation fold.
    resampling : dict
        Dictionary containing dataset frequency information.

    Returns
    -------
    pd.DataFrame
        Dataframe that will be used to plot cross-validation folds with plotly.
    str
        Hover template that will be used to show cross-validation folds dates on a plotly viz.
    """
    hover_data = pd.DataFrame(cv_dates).T
    if resampling["freq"][-1] in ["s", "H"]:
        hover_data = hover_data.applymap(lambda x: x.strftime("%Y/%m/%d %H:%M:%S"))
    else:
        hover_data = hover_data.applymap(lambda x: x.strftime("%Y/%m/%d"))
    hover_template = "<br>".join(
        [
            "%{y}",
            "Training start date: %{text[0]}",
            "Training end date: %{text[2]}",
            "Validation start date: %{text[1]}",
            "Validation end date: %{text[3]}",
        ]
    )
    return hover_data, hover_template
