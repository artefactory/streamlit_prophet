from datetime import timedelta

import pandas as pd
from streamlit_prophet.lib.utils.mapping import convert_into_nb_of_days, convert_into_nb_of_seconds


def get_forecast_components(models: dict, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a dataframe with only the relevant components to sum to get the prediction
    """
    fcst = forecast_df.copy()
    components_col_names = get_forecast_components_col_names(fcst) + ["ds"]
    components = fcst[components_col_names]
    for col in components_col_names:
        if col in models["eval"].component_modes["multiplicative"]:
            components[col] *= components["trend"]
    components = components.set_index("ds")
    return components


def get_forecast_components_col_names(forecast: pd.DataFrame) -> list:
    components_col = [
        col.replace("_lower", "")
        for col in forecast.columns
        if "lower" in col
        and "yhat" not in col
        and "multiplicative" not in col
        and "additive" not in col
    ]
    return components_col


def get_df_cv_with_hist(forecasts: dict, datasets: dict, models: dict) -> pd.DataFrame:
    df_cv = forecasts["cv"].drop(["cutoff"], axis=1)
    df_past = models["eval"].predict(
        datasets["train"].loc[datasets["train"]["ds"] < df_cv.ds.min()].drop("y", axis=1)
    )
    common_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    df_past = df_past[common_cols + list(set(df_past.columns) - set(common_cols))]
    df_cv = pd.concat([df_cv, df_past], axis=0).sort_values("ds").reset_index(drop=True)
    return df_cv


def get_cv_dates_dict(dates: dict, resampling: dict) -> dict:
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


def get_hover_template_cv(cv_dates: dict, resampling: dict):
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
