from pathlib import Path
import toml
from loguru import logger
import streamlit as st
import pandas as pd
from fbprophet.plot import plot_components_plotly, plot_plotly

from lib.utils.path import get_project_root
from lib.utils.logging import suppress_stdout_stderr

from lib.dataprep.clean import format_columns, clean_timeseries
from lib.dataprep.split import train_val_split, make_eval_df, make_future_df

from lib.inputs.dataset import input_dataset, input_columns
from lib.inputs.dataprep import input_cleaning
from lib.inputs.dates import input_split_dates, input_cv_dates, input_forecast_dates
from lib.inputs.params import (input_prior_scale_params,
                               input_seasonality_params,
                               input_holidays_params,
                               input_other_params
                               )

from lib.models.prophet import get_prophet_model

from lib.evaluation.preparation import get_evaluation_series, get_evaluation_df
from lib.evaluation.metrics import get_perf_metrics, prettify_metrics

from lib.exposition.visualize import plot_forecasts_vs_truth, plot_truth_vs_actual_scatter, plot_residuals_distrib

config = toml.load(Path(get_project_root()) / f'config/config_streamlit.toml')

st.sidebar.title("1. Data")

# Load data
with st.sidebar.beta_expander("Select a dataset", expanded=False):
    df = input_dataset()

# Column names
with st.sidebar.beta_expander("Columns", expanded=False):
    date_col, target_col = input_columns(config)
    df = format_columns(df, date_col, target_col)

# Filtering
with st.sidebar.beta_expander("Filtering", expanded=False):
    """Ajouter section filtering"""

# Cleaning
with st.sidebar.beta_expander("Cleaning", expanded=False):
    del_days, del_zeros, del_negative = input_cleaning()
    df = clean_timeseries(df, del_negative, del_zeros, del_days)

# Evaluation process
with st.sidebar.beta_expander("Evaluation process", expanded=False):
    use_cv = st.checkbox("Perform cross-validation", value=False)
    dates = dict()
    if use_cv:
        dates = input_cv_dates(df, dates)
        # TODO: Implémenter st.success des dates de chaque fold de cross-val
        logger.warning("CV not implemented yet")
    else:
        dates = input_split_dates(df, dates)
        #TODO : mettre train, val, cv, future et future_eval dans un dict datasets
        train, val = train_val_split(df, dates)
        st.success(
            f"""Train: {train.ds.min().strftime('%d/%m/%Y')} - {train.ds.max().strftime('%d/%m/%Y')}
                Valid: {val.ds.min().strftime('%d/%m/%Y')} - {val.ds.max().strftime('%d/%m/%Y')} 
                ({round((len(val) / float(len(df)) * 100))}% of data used for validation)""")

# Forecast
with st.sidebar.beta_expander("Forecast", expanded=False):
    make_future_forecast = st.checkbox("Make forecast on future dates", value=False)
    if make_future_forecast:
        dates = input_forecast_dates(df, dates, config)
        st.success(
            f"""Forecast: {dates['forecast_start_date'].strftime('%d/%m/%Y')} - 
            {dates['forecast_end_date'].strftime('%d/%m/%Y')}""")

st.sidebar.title("2. Modelling")

# Prior scale
with st.sidebar.beta_expander("Prior scale", expanded=False):
    """Increase values to make it more flexible"""
    prior_scale_params = input_prior_scale_params(config)

# Seasonalities
with st.sidebar.beta_expander("Seasonalities", expanded=False):
    seasonalities = input_seasonality_params()

# Holidays and events
with st.sidebar.beta_expander("Holidays and events"):
    holidays = input_holidays_params(config)
    #TODO: Ajouter la possibilité d'entrer une date d'événement à encoder (ex: confinemnent)

# External regressors
with st.sidebar.beta_expander("External regressors"):
    """Implémenter l'ajout de régresseurs"""

# Other parameters
with st.sidebar.beta_expander("Other parameters", expanded=False):
    other_params = input_other_params(config)

# Train & Forecast
if st.checkbox('Relaunch forecast automatically when parameters change', value=True):
    launch_forecast = True
else:
    launch_forecast = st.button('Launch forecast')
if launch_forecast:
    #TODO: Mettre ce bloc dans une fonction dans models
    if use_cv:
        #TODO: Implémenter cross-val
        logger.warning("CV not implemented yet")
    else:
        model_eval = get_prophet_model(prior_scale_params, other_params, seasonalities, holidays)
        with suppress_stdout_stderr():
            model_eval.fit(train, seed=config["global"]["seed"])
        future_eval = make_eval_df(train, val)
        forecast_eval = model_eval.predict(future_eval)
    if make_future_forecast:
        model = get_prophet_model(prior_scale_params, other_params, seasonalities, holidays)
        with suppress_stdout_stderr():
            model.fit(pd.concat([train, val], axis=0), seed=config["global"]["seed"])
        future = make_future_df(df, dates, include_history=True)
        #TODO : Appliquer le même cleaning / les mêmes filtres sur la donnée future que sur l'historique
        forecast = model.predict(future)


st.sidebar.title("3. Evaluation")

# Performance metrics
with st.sidebar.beta_expander("Metrics", expanded=False):
    metrics = st.multiselect("Choose evaluation metrics",
                             config["evaluation"]["metrics"],
                             default=['MAPE', 'RMSE']
                             )

with st.sidebar.beta_expander("Scope", expanded=False):
    eval_set = st.selectbox("Choose evaluation set", ['Validation', 'Training'])
    """Implémenter granularité d'éval"""

st.write('# 1. Overview')
#TODO: Implémenter fonction get_overview plots
#TODO: Mettre model, forecast, model_eval et forecast_eval dans un dict
if make_future_forecast:
    st.plotly_chart(plot_plotly(model, forecast, changepoints=True, trend=True))
    st.plotly_chart(plot_components_plotly(model, forecast))
else:
    st.plotly_chart(plot_plotly(model_eval, forecast_eval, changepoints=True, trend=True))
    st.plotly_chart(plot_components_plotly(model_eval, forecast_eval))

st.write(f'# 2. Model performance on {eval_set.lower()} set')

if use_cv:
    # TODO: Implémenter cross-val evaluation
    logger.warning("CV not implemented yet")
else:
    #TODO: Refacto pour appeler directement eval_df dans get_perf_metrics
    y_true, y_pred = get_evaluation_series(train, val, forecast_eval, dates, eval_set)
    eval_df = get_evaluation_df(train, val, forecast_eval, dates, eval_set)
    perf = get_perf_metrics(y_true, y_pred, metrics)
    st.success(prettify_metrics(perf))
    st.plotly_chart(plot_forecasts_vs_truth(eval_df, target_col))
    st.plotly_chart(plot_truth_vs_actual_scatter(eval_df))
    st.plotly_chart(plot_residuals_distrib(eval_df))

#st.write('# 3. Impact of components and regressors')