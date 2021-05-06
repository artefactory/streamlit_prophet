# type: ignore

import pandas as pd
import streamlit as st
from streamlit_prophet.lib.utils.mapping import mapping_country_names


def input_seasonality_params(config: dict, params: dict, resampling: dict, readme: dict) -> dict:
    default_params = config["model"]
    seasonalities = {
        "yearly": {"period": 365.25, "prophet_param": None},
        "monthly": {"period": 30.5, "prophet_param": None},
        "weekly": {"period": 7, "prophet_param": None},
    }
    if resampling["freq"][-1] in ["s", "H"]:
        seasonalities["daily"] = {"period": 1, "prophet_param": None}
    for seasonality, values in seasonalities.items():
        values["prophet_param"] = st.selectbox(
            f"{seasonality.capitalize()} seasonality",
            ["auto", False, "custom"] if seasonality[0] in ["y", "w", "d"] else [False, "custom"],
            help=readme["tooltips"]["seasonality"],
        )
        if values["prophet_param"] == "custom":
            values["prophet_param"] = False
            values["custom_param"] = {
                "name": seasonality,
                "period": values["period"],
                "mode": st.selectbox(
                    f"Seasonality mode for {seasonality} seasonality",
                    default_params["seasonality_mode"],
                    help=readme["tooltips"]["seasonality_mode"],
                ),
                "fourier_order": st.number_input(
                    f"Fourier order for {seasonality} seasonality",
                    value=15,
                    help=readme["tooltips"]["seasonality_fourier"],
                ),
                "prior_scale": st.number_input(
                    f"Prior scale for {seasonality} seasonality",
                    value=10,
                    help=readme["tooltips"]["seasonality_prior_scale"],
                ),
            }
    add_custom_seasonality = st.checkbox(
        "Add a custom seasonality", value=False, help=readme["tooltips"]["add_custom_seasonality"]
    )
    if add_custom_seasonality:
        custom_seasonality = dict()
        custom_seasonality["custom_param"] = dict()
        custom_seasonality["custom_param"]["name"] = st.text_input(
            "Name", value="custom_seasonality", help=readme["tooltips"]["seasonality_name"]
        )
        custom_seasonality["custom_param"]["period"] = st.number_input(
            "Period (in days)", value=10, help=readme["tooltips"]["seasonality_period"]
        )
        custom_seasonality["custom_param"]["mode"] = st.selectbox(
            f"Mode", default_params["seasonality_mode"], help=readme["tooltips"]["seasonality_mode"]
        )
        custom_seasonality["custom_param"]["fourier_order"] = st.number_input(
            f"Fourier order", value=15, help=readme["tooltips"]["seasonality_fourier"]
        )
        custom_seasonality["custom_param"]["prior_scale"] = st.number_input(
            f"Prior scale", value=10, help=readme["tooltips"]["seasonality_prior_scale"]
        )
        seasonalities[custom_seasonality["custom_param"]["name"]] = custom_seasonality
    params["seasonalities"] = seasonalities
    return params


def input_prior_scale_params(config: dict, readme: dict) -> dict:
    params = dict()
    default_params = config["model"]
    seasonality_prior_scale = st.number_input(
        "seasonality_prior_scale",
        value=default_params["seasonality_prior_scale"],
        help=readme["tooltips"]["seasonality_prior_scale"],
    )
    holidays_prior_scale = st.number_input(
        "holidays_prior_scale",
        value=default_params["holidays_prior_scale"],
        help=readme["tooltips"]["holidays_prior_scale"],
    )
    changepoint_prior_scale = st.number_input(
        "changepoint_prior_scale",
        value=default_params["changepoint_prior_scale"],
        format="%.3f",
        help=readme["tooltips"]["changepoint_prior_scale"],
    )
    params["prior_scale"] = {
        "seasonality_prior_scale": seasonality_prior_scale,
        "holidays_prior_scale": holidays_prior_scale,
        "changepoint_prior_scale": changepoint_prior_scale,
    }
    return params


def input_other_params(config: dict, params: dict, readme: dict) -> dict:
    default_params = config["model"]
    growth = st.selectbox("growth", default_params["growth"], help=readme["tooltips"]["growth"])
    n_changepoints = st.number_input(
        "n_changepoints",
        value=default_params["n_changepoints"],
        help=readme["tooltips"]["n_changepoints"],
    )
    changepoint_range = st.number_input(
        "changepoint_range",
        value=default_params["changepoint_range"],
        format="%.2f",
        help=readme["tooltips"]["changepoint_range"],
    )
    params["other"] = {
        "growth": growth,
        "n_changepoints": n_changepoints,
        "changepoint_range": changepoint_range,
    }
    # TODO: Régler l'erreur avec growth = 'logistic'
    return params


def input_holidays_params(params: dict, readme: dict) -> dict:
    countries = sorted(mapping_country_names([])[0].keys())
    params["holidays"] = st.multiselect(
        "Add some countries' holidays", countries, default=[], help=readme["tooltips"]["holidays"]
    )
    _, params["holidays"] = mapping_country_names(params["holidays"])
    return params


def input_regressors(df: pd.DataFrame, config: dict, params: dict, readme: dict) -> dict:
    regressors = dict()
    default_params = config["model"]
    all_cols = set(df.columns) - {"ds", "y"}
    mask = df[all_cols].isnull().sum() == 0
    eligible_cols = list(mask[mask].index)
    _print_removed_regressors(list(set(all_cols) - set(eligible_cols)))
    if len(eligible_cols) > 0:
        if st.checkbox(
            "Add all detected regressors",
            value=False,
            help=readme["tooltips"]["add_all_regressors"],
        ):
            default_regressors = list(eligible_cols)
        else:
            default_regressors = []
        regressor_cols = st.multiselect(
            "Select external regressors if any",
            list(eligible_cols),
            default=default_regressors,
            help=readme["tooltips"]["select_regressors"],
        )
        for col in regressor_cols:
            regressors[col] = dict()
            regressors[col]["prior_scale"] = st.number_input(
                f"Prior scale for {col}",
                value=default_params["regressors_prior_scale"],
                help=readme["tooltips"]["regressor_mode"],
            )
            regressors[col]["mode"] = st.selectbox(
                f"Mode for {col}",
                default_params["seasonality_mode"],
                help=readme["tooltips"]["regressor_prior_scale"],
            )
    else:
        st.write("There are no regressors in your dataset.")
    params["regressors"] = regressors
    return params


def _print_removed_regressors(nan_cols: list):
    L = len(nan_cols)
    if L > 0:
        st.error(
            f'The following column{"s" if L > 1 else ""} cannot be taken as regressor because '
            f'{"they contain" if L > 1 else "it contains"} null values: {", ".join(nan_cols)}'
        )
