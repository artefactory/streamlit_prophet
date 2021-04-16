from pathlib import Path
import toml
from loguru import logger
import streamlit as st
from fbprophet.plot import plot_components_plotly, plot_plotly

from lib.utils.load import initialisation

from lib.dataprep.clean import format_columns, clean_timeseries
from lib.dataprep.split import train_val_split

from lib.inputs.dataset import input_dataset, input_columns
from lib.inputs.dataprep import input_cleaning
from lib.inputs.dates import input_split_dates, input_cv_dates, input_forecast_dates
from lib.inputs.params import (input_prior_scale_params,
                               input_seasonality_params,
                               input_holidays_params,
                               input_other_params
                               )

from lib.models.prophet import forecast_worklow

from lib.evaluation.preparation import get_evaluation_series, get_evaluation_df
from lib.evaluation.metrics import get_perf_metrics, prettify_metrics

from lib.exposition.visualize import plot_forecasts_vs_truth, plot_truth_vs_actual_scatter, plot_residuals_distrib

config, params, dates, datasets, models, forecasts = initialisation('config_streamlit.toml')

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
    if use_cv:
        dates = input_cv_dates(df, dates)
        # TODO: Implémenter st.success des dates de chaque fold de cross-val
        logger.warning("CV not implemented yet")
    else:
        dates = input_split_dates(df, dates)
        datasets = train_val_split(df, dates, datasets)
        st.success(
            f"""Train: {datasets['train'].ds.min().strftime('%d/%m/%Y')} - 
                       {datasets['train'].ds.max().strftime('%d/%m/%Y')}
                Valid: {datasets['val'].ds.min().strftime('%d/%m/%Y')} - 
                       {datasets['val'].ds.max().strftime('%d/%m/%Y')} 
                ({round((len(datasets['val']) / float(len(df)) * 100))}% of data used for validation)""")

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
    params = input_prior_scale_params(config, params)

# Seasonalities
with st.sidebar.beta_expander("Seasonalities", expanded=False):
    params = input_seasonality_params(params)

# Holidays and events
with st.sidebar.beta_expander("Holidays and events"):
    params = input_holidays_params(config, params)
    #TODO: Ajouter la possibilité d'entrer une date d'événement à encoder (ex: confinemnent)

# External regressors
with st.sidebar.beta_expander("External regressors"):
    """Implémenter l'ajout de régresseurs"""

# Other parameters
with st.sidebar.beta_expander("Other parameters", expanded=False):
    params = input_other_params(config, params)

# Train & Forecast
if st.checkbox('Relaunch forecast automatically when parameters change', value=True):
    launch_forecast = True
else:
    launch_forecast = st.button('Launch forecast')
if launch_forecast:
    datasets, models, forecasts = forecast_worklow(config, use_cv, make_future_forecast,
                                                   df, params, dates, datasets, models, forecasts)


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
if make_future_forecast:
    st.plotly_chart(plot_plotly(models['future'], forecasts['future'], changepoints=True, trend=True))
    st.plotly_chart(plot_components_plotly(models['future'], forecasts['future']))
else:
    st.plotly_chart(plot_plotly(models['eval'], forecasts['eval'], changepoints=True, trend=True))
    st.plotly_chart(plot_components_plotly(models['eval'], forecasts['eval']))

st.write(f'# 2. Model performance on {eval_set.lower()} set')

if use_cv:
    # TODO: Implémenter cross-val evaluation
    logger.warning("CV not implemented yet")
else:
    #TODO: Refacto pour appeler directement eval_df dans get_perf_metrics
    #TODO: Refacto pour appeler datasets et forecasts dict uniquement
    y_true, y_pred = get_evaluation_series(datasets['train'], datasets['val'], forecasts['eval'], dates, eval_set)
    eval_df = get_evaluation_df(datasets['train'], datasets['val'], forecasts['eval'], dates, eval_set)
    perf = get_perf_metrics(y_true, y_pred, metrics)
    st.success(prettify_metrics(perf))
    st.plotly_chart(plot_forecasts_vs_truth(eval_df, target_col))
    st.plotly_chart(plot_truth_vs_actual_scatter(eval_df))
    st.plotly_chart(plot_residuals_distrib(eval_df))

#st.write('# 3. Impact of components and regressors')