from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from streamlit_prophet.lib.utils.holidays import lockdown_format_func
from streamlit_prophet.lib.utils.mapping import (
    COUNTRY_NAMES_MAPPING,
    COVID_LOCKDOWN_DATES_MAPPING,
    SCHOOL_HOLIDAYS_FUNC_MAPPING,
)


def input_seasonality_params(
    config: Dict[Any, Any],
    params: Dict[Any, Any],
    resampling: Dict[Any, Any],
    readme: Dict[Any, Any],
) -> Dict[Any, Any]:
    """Lets the user enter seasonality parameters.

    Parameters
    ----------
    params : Dict
        Model parameters.
    config : Dict
        Lib config dictionary containing information about default parameters.
    resampling : Dict
        Dictionary containing dataset frequency information.
    readme : Dict
        Dictionary containing tooltips to guide user's choices.

    Returns
    -------
    dict
        Model parameters with seasonality parameters added.
    """
    default_params = config["model"]
    seasonalities: Dict[str, Dict[Any, Any]] = {
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
        custom_seasonality: Dict[Any, Any] = dict()
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


def input_prior_scale_params(config: Dict[Any, Any], readme: Dict[Any, Any]) -> Dict[Any, Any]:
    """Lets the user enter prior scale parameters.

    Parameters
    ----------
    config : Dict
        Lib config dictionary containing information about default parameters.
    readme : Dict
        Dictionary containing tooltips to guide user's choices.

    Returns
    -------
    dict
        Model prior scale parameters.
    """
    params = dict()
    default_params = config["model"]
    changepoint_prior_scale = st.number_input(
        "changepoint_prior_scale",
        value=default_params["changepoint_prior_scale"],
        format="%.3f",
        help=readme["tooltips"]["changepoint_prior_scale"],
    )
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
    params["prior_scale"] = {
        "seasonality_prior_scale": seasonality_prior_scale,
        "holidays_prior_scale": holidays_prior_scale,
        "changepoint_prior_scale": changepoint_prior_scale,
    }
    return params


def input_other_params(
    config: Dict[Any, Any], params: Dict[Any, Any], readme: Dict[Any, Any]
) -> Dict[Any, Any]:
    """Lets the user enter other parameters (growth, changepoints_range, n_changepoints).

    Parameters
    ----------
    config : Dict
        Lib config dictionary containing information about default parameters.
    params : Dict
        Model parameters.
    readme : Dict
        Dictionary containing tooltips to guide user's choices.

    Returns
    -------
    dict
        Model parameters with other parameters added.
    """
    default_params = config["model"]
    changepoint_range = st.number_input(
        "changepoint_range",
        value=default_params["changepoint_range"],
        max_value=1.0,
        min_value=0.0,
        format="%.2f",
        help=readme["tooltips"]["changepoint_range"],
    )
    growth = st.selectbox("growth", default_params["growth"], help=readme["tooltips"]["growth"])
    params["other"] = {
        "growth": growth,
        "changepoint_range": changepoint_range,
    }
    if growth == "logistic":
        cap = st.number_input(
            "cap",
            value=default_params["cap"],
            format="%.1f",
            help=readme["tooltips"]["cap"],
        )
        floor = st.number_input(
            "floor",
            value=default_params["floor"],
            format="%.1f",
            max_value=cap,
            help=readme["tooltips"]["floor"],
        )
        params["saturation"] = {
            "cap": cap,
            "floor": floor,
        }
    return params


def input_holidays_params(
    params: Dict[Any, Any], readme: Dict[Any, Any], config: Dict[Any, Any]
) -> Dict[Any, Any]:
    """Lets the user enter holidays parameters.

    Parameters
    ----------
    params : Dict
        Model parameters.
    readme : Dict
        Dictionary containing tooltips to guide user's choices.
    config : Dict
        Dictionary where user can provide the list of countries whose holidays will be included.

    Returns
    -------
    dict
        Model parameters with holidays parameters added.
    """
    countries = list(COUNTRY_NAMES_MAPPING.keys())
    default_country = config["model"]["holidays_country"]
    country = st.selectbox(
        label="Select a country",
        options=countries,
        index=countries.index(default_country),
        format_func=lambda x: COUNTRY_NAMES_MAPPING[x],
        help=readme["tooltips"]["holidays_country"],
    )

    public_holidays = st.checkbox(
        label="Public holidays",
        value=config["model"]["public_holidays"],
        help=readme["tooltips"]["public_holidays"],
    )

    school_holidays = False
    if country in SCHOOL_HOLIDAYS_FUNC_MAPPING.keys():
        school_holidays = st.checkbox(
            label="School holidays",
            value=config["model"]["school_holidays"],
            help=readme["tooltips"]["school_holidays"],
        )

    lockdowns = []
    if country in COVID_LOCKDOWN_DATES_MAPPING.keys():
        lockdown_options = list(range(len(COVID_LOCKDOWN_DATES_MAPPING[country])))
        lockdowns = st.multiselect(
            label="Lockdown events",
            options=lockdown_options,
            default=config["model"]["lockdown_events"],
            format_func=lockdown_format_func,
            help=readme["tooltips"]["lockdown_events"],
        )

    params["holidays"] = {
        "country": country,
        "public_holidays": public_holidays,
        "school_holidays": school_holidays,
        "lockdown_events": lockdowns,
    }
    return params


def input_regressors(
    df: pd.DataFrame, config: Dict[Any, Any], params: Dict[Any, Any], readme: Dict[Any, Any]
) -> Dict[Any, Any]:
    """Lets the user select regressors.

    Parameters
    ----------
    df : Dict
        Prepared dataset (after filtering, resampling, cleaning).
    config : Dict
        Lib config dictionary containing information about default parameters.
    params : Dict
        Model parameters.
    readme : Dict
        Dictionary containing tooltips to guide user's choices.

    Returns
    -------
    dict
        Model parameters with regressors information added.
    """
    regressors: Dict[Any, Any] = dict()
    default_params = config["model"]
    all_cols = set(df.columns) - {"ds", "y"}
    mask = df[all_cols].isnull().sum() == 0
    eligible_cols = sorted(list(mask[mask].index))
    _print_removed_regressors(sorted(set(all_cols) - set(eligible_cols)))
    if len(eligible_cols) > 0:
        if st.checkbox(
            "Add all detected regressors",
            value=False,
            help=readme["tooltips"]["add_all_regressors"],
        ):
            default_regressors = list(eligible_cols)
        else:
            default_regressors = []
        config_regressors = config["columns"]["regressors"]
        if config_regressors not in ["false", False]:
            if len(set(config_regressors).intersection(set(eligible_cols))) != len(
                config_regressors
            ):
                st.error(
                    f"Selected regressors are not in the dataset columns, "
                    f"please provide a list of valid columns for regressors in the config file."
                )
                st.stop()
        regressor_cols = st.multiselect(
            "Select external regressors if any",
            list(eligible_cols),
            default=default_regressors
            if config_regressors in ["false", False]
            else config_regressors,
            help=readme["tooltips"]["select_regressors"],
        )
        for col in regressor_cols:
            regressors[col] = dict()
            regressors[col]["prior_scale"] = st.number_input(
                f"Prior scale for {col}",
                value=default_params["regressors_prior_scale"],
                help=readme["tooltips"]["regressor_prior_scale"],
            )
    else:
        st.write("There are no regressors in your dataset.")
    params["regressors"] = regressors
    return params


def _print_removed_regressors(nan_cols: List[Any]) -> None:
    """Displays a message in streamlit dashboard if the input list is not empty.

    Parameters
    ----------
    nan_cols : list
        List of columns that have been removed because they contain null values.
    """
    L = len(nan_cols)
    if L > 0:
        st.error(
            f'The following column{"s" if L > 1 else ""} cannot be taken as regressor because '
            f'{"they contain" if L > 1 else "it contains"} null values: {", ".join(nan_cols)}'
        )
