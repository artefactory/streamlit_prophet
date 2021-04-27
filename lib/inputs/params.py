import streamlit as st
import pandas as pd
from lib.utils.mapping import mapping_country_names


def input_seasonality_params(config: dict, params: dict, resampling: dict) -> dict:
    default_params = config["model"]
    seasonalities = {'yearly': {'period': 365.25, 'prophet_param': None},
                     'monthly': {'period': 30.5, 'prophet_param': None},
                     'weekly': {'period': 7, 'prophet_param': None}
                     }
    if resampling['freq'][-1] in ['s', 'H']:
        seasonalities['daily'] = {'period': 1, 'prophet_param': None}
    for seasonality, values in seasonalities.items():
        values['prophet_param'] = st.selectbox(
            f"{seasonality.capitalize()} seasonality", ['auto', False, 'custom'])
        if values['prophet_param'] == 'custom':
            values['prophet_param'] = False
            values['custom_param'] = {
                'name': seasonality,
                'period': values['period'],
                'mode': st.selectbox(f"Seasonality mode for {seasonality} seasonality",
                                     default_params['seasonality_mode']),
                'fourier_order': st.number_input(f"Fourier order for {seasonality} seasonality", value=15),
                'prior_scale': st.number_input(f"Prior scale for {seasonality} seasonality", value=10.0),
            }
    add_custom_seasonality = st.checkbox("Add a custom seasonality", value=False)
    if add_custom_seasonality:
        custom_seasonality = dict()
        custom_seasonality['custom_param'] = dict()
        custom_seasonality['custom_param']['name'] = st.text_input("Name", value='custom_seasonality')
        custom_seasonality['custom_param']['period'] = st.number_input("Period (in days)", value=10)
        custom_seasonality['custom_param']['mode'] = st.selectbox(f"Mode", default_params['seasonality_mode'])
        custom_seasonality['custom_param']['fourier_order'] = st.number_input(f"Fourier order", value=15)
        custom_seasonality['custom_param']['prior_scale'] = st.number_input(f"Prior scale", value=10.0)
        seasonalities[custom_seasonality['custom_param']['name']] = custom_seasonality
    params['seasonalities'] = seasonalities
    return params


def input_prior_scale_params(config: dict) -> dict:
    params = dict()
    default_params = config["model"]
    seasonality_prior_scale = st.number_input("seasonality_prior_scale",
                                              value=default_params['seasonality_prior_scale'], )
    holidays_prior_scale = st.number_input("holidays_prior_scale",
                                           value=default_params['holidays_prior_scale'], )
    changepoint_prior_scale = st.number_input("changepoint_prior_scale",
                                              value=default_params['changepoint_prior_scale'], )
    params['prior_scale'] = {'seasonality_prior_scale': seasonality_prior_scale,
                             'holidays_prior_scale': holidays_prior_scale,
                             'changepoint_prior_scale': changepoint_prior_scale
                             }
    return params


def input_other_params(config: dict, params: dict) -> dict:
    default_params = config["model"]
    growth = st.selectbox("growth", default_params['growth'])
    #seasonality_mode = st.selectbox("seasonality_mode", default_params['seasonality_mode'])
    n_changepoints = st.number_input("n_changepoints", value=default_params['n_changepoints'])
    changepoint_range = st.number_input("changepoint_range", value=default_params['changepoint_range'])
    params['other'] = {'growth': growth,
                       #'seasonality_mode': seasonality_mode,
                       'n_changepoints': n_changepoints,
                       'changepoint_range': changepoint_range
                       }
    # TODO: Régler l'erreur avec growth = 'logistic'
    return params


def input_holidays_params(params: dict) -> dict:
    countries = sorted(mapping_country_names([])[0].keys())
    params['holidays'] = st.multiselect("Add some countries' holidays", countries, default=[])
    _, params['holidays'] = mapping_country_names(params['holidays'])
    return params


def input_regressors(df: pd.DataFrame, config: dict, params: dict) -> dict:
    regressors = dict()
    default_params = config["model"]
    eligible_cols = set(df.columns) - set(['ds', 'y'])
    if len(eligible_cols) > 0:
        if st.checkbox('Add all detected regressors', value=False):
            default_regressors = list(eligible_cols)
        else:
            default_regressors = []
        regressor_cols = st.multiselect("Select external regressors if any",
                                        list(eligible_cols),
                                        default=default_regressors
                                        )
        for col in regressor_cols:
            regressors[col] = dict()
            regressors[col]['prior_scale'] = st.number_input(f"prior_scale for {col}",
                                                             value=default_params['regressors_prior_scale'])
            regressors[col]['mode'] = st.selectbox(f"mode for {col}", default_params['seasonality_mode'])
    else:
        st.write("There are no regressors in your dataset.")
    params['regressors'] = regressors
    return params
