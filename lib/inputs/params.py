import streamlit as st


def input_seasonality_params(params):
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
    params['seasonalities'] = seasonalities
    return params


def input_prior_scale_params(config, params):
    default_params = config["model"]["input_params"]
    seasonality_prior_scale = st.number_input("seasonality_prior_scale",
                                              value=default_params['seasonality_prior_scale'], )
    holidays_prior_scale = st.number_input("holidays_prior_scale:",
                                           value=default_params['holidays_prior_scale'], )
    changepoint_prior_scale = st.number_input("changepoint_prior_scale:",
                                              value=default_params['changepoint_prior_scale'], )
    params['prior_scale'] = {'seasonality_prior_scale': seasonality_prior_scale,
                             'holidays_prior_scale': holidays_prior_scale,
                             'changepoint_prior_scale': changepoint_prior_scale
                             }
    return params


def input_other_params(config, params):
    default_params = config["model"]["input_params"]
    growth = st.selectbox("growth", default_params['growth'])
    seasonality_mode = st.selectbox("seasonality_mode", default_params['seasonality_mode'])
    n_changepoints = st.number_input("n_changepoints", value=default_params['n_changepoints'])
    changepoint_range = st.number_input("changepoint_range", value=default_params['changepoint_range'])
    params['other'] = {'growth': growth,
                       'seasonality_mode': seasonality_mode,
                       'n_changepoints': n_changepoints,
                       'changepoint_range': changepoint_range
                       }
    return params


def input_holidays_params(config, params):
    countries = config["model"]["input_params"]["holidays"]
    # TODO: Ajouter un mapping pour avoir les noms de pays bien écrits
    params['holidays'] = st.multiselect("Add some countries' holidays", countries, default=[])
    return params


def input_regressors(df, config, params):
    regressors = dict()
    default_params = config["model"]["input_params"]
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