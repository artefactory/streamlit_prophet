from pathlib import Path
import streamlit as st
import toml
from lib.utils.path import get_project_root

def input_seasonality_params():
    seasonalities = {
        'yearly': {
            'period': 365.25,
            'prophet_param': None,
            },
        'monthly': {
            'period': 30.5,
            'prophet_param': None,
            },
        'weekly': {
            'period': 7,
            'prophet_param': None,
            },
        }
    for seasonality, values in seasonalities.items():
        values['prophet_param'] = st.selectbox(
            f"{seasonality}_seasonality", ['auto', False, 'custom'])
        if values['prophet_param'] == 'custom':
            values['prophet_param'] = False
            values['custom_param'] = {
                'name': seasonality,
                'period': values['period'],
                'fourier_order': st.number_input(f"fourrier_order_{seasonality}", value=15),
                'prior_scale': st.number_input(f"prior_scale_{seasonality}", value=10.0),
            }
    return seasonalities


def input_prior_scale_params(config):
    params = config["model"]["input_params"]
    seasonality_prior_scale = st.number_input("seasonality_prior_scale", value=params['seasonality_prior_scale'], )
    holidays_prior_scale = st.number_input("holidays_prior_scale:", value=params['holidays_prior_scale'], )
    changepoint_prior_scale = st.number_input("changepoint_prior_scale:", value=params['changepoint_prior_scale'], )
    return {'seasonality_prior_scale': seasonality_prior_scale,
            'holidays_prior_scale': holidays_prior_scale,
            'changepoint_prior_scale': changepoint_prior_scale
            }

def input_other_params(config):
    params = config["model"]["input_params"]
    growth = st.selectbox("growth", params['growth'])
    seasonality_mode = st.selectbox("seasonality_mode", params['seasonality_mode'])
    n_changepoints = st.number_input("n_changepoints", value=params['n_changepoints'])
    changepoint_range = st.number_input("changepoint_range", value=params['changepoint_range'])
    return {'growth': growth,
            'seasonality_mode': seasonality_mode,
            'n_changepoints': n_changepoints,
            'changepoint_range': changepoint_range
            }


def input_holidays_params(config):
    countries = config["model"]["input_params"]["holidays"]
    #TODO: Ajouter un mapping pour avoir les noms de pays bien écrits
    holidays = st.multiselect("Add some countries' holidays", countries, default=[])
    return holidays