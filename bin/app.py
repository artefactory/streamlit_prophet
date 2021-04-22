import streamlit as st
from lib.utils.load import load_config
from lib.dataprep.clean import format_date_and_target, clean_df
from lib.dataprep.format import filter_and_aggregate_df, resample_df
from lib.dataprep.split import get_train_val_sets, get_train_set
from lib.inputs.dataset import input_dataset, input_columns
from lib.inputs.dataprep import input_dimensions, input_resampling, input_cleaning
from lib.inputs.dates import input_train_dates, input_val_dates, input_cv, input_forecast_dates
from lib.inputs.params import (input_prior_scale_params,
                               input_seasonality_params,
                               input_holidays_params,
                               input_other_params,
                               input_regressors
                               )
from lib.inputs.eval import input_metrics, input_scope_eval
from lib.models.prophet import forecast_workflow
from lib.exposition.visualize import plot_overview, plot_performance, plot_components

# Initialization
config, readme = load_config('config_streamlit.toml', 'config_readme.toml')
params, cleaning, dates, datasets, models, forecasts, eval = dict(), dict(), dict(), dict(), dict(), dict(), dict()

st.sidebar.title("1. Data")

# Load data
with st.sidebar.beta_expander("Dataset", expanded=True):
    df, load_options = input_dataset(config)

# Column names
with st.sidebar.beta_expander("Columns", expanded=True):
    date_col, target_col = input_columns(config, df, load_options)
    df = format_date_and_target(df, date_col, target_col)

# Filtering
with st.sidebar.beta_expander("Filtering", expanded=False):
    dimensions = input_dimensions(df)
    df = filter_and_aggregate_df(df, dimensions)

# Resampling
with st.sidebar.beta_expander("Resampling", expanded=False):
    resampling = input_resampling(df)
    if resampling['resample']:
        df = resample_df(df, resampling)

# Cleaning
with st.sidebar.beta_expander("Cleaning", expanded=False):
    cleaning = input_cleaning(cleaning, resampling)
    df = clean_df(df, cleaning)

# Evaluation process
with st.sidebar.beta_expander("Evaluation process", expanded=False):
    use_cv = st.checkbox("Perform cross-validation", value=False)
    dates = input_train_dates(df, dates, use_cv)
    if use_cv:
        dates = input_cv(dates)
        datasets = get_train_set(df, dates, datasets)
    else:
        dates = input_val_dates(df, dates)
        datasets = get_train_val_sets(df, dates, datasets)

# Forecast
with st.sidebar.beta_expander("Forecast", expanded=False):
    make_future_forecast = st.checkbox("Make forecast on future dates", value=False)
    if make_future_forecast:
        dates = input_forecast_dates(df, dates, resampling)

st.sidebar.title("2. Modelling")

# Prior scale
with st.sidebar.beta_expander("Prior scale", expanded=False):
    params = input_prior_scale_params(config, params)

# Seasonalities
with st.sidebar.beta_expander("Seasonalities", expanded=False):
    params = input_seasonality_params(params)

# Holidays
with st.sidebar.beta_expander("Holidays"):
    params = input_holidays_params(params)

# External regressors
with st.sidebar.beta_expander("External regressors"):
    params = input_regressors(df, config, params)

# Other parameters
with st.sidebar.beta_expander("Other parameters", expanded=False):
    params = input_other_params(config, params)

st.sidebar.title("3. Evaluation")

# Performance metrics
with st.sidebar.beta_expander("Metrics", expanded=False):
    eval = input_metrics(eval)

# Scope of evaluation
with st.sidebar.beta_expander("Scope", expanded=False):
    eval = input_scope_eval(eval)

# Info
with st.beta_expander("What is this app ?", expanded=False):
    st.write(readme['app']['app_intro'])
with st.beta_expander("More info on model parameters", expanded=False):
    st.write(readme['params']['prophet_params'])
st.write('')

# Launch training & forecast
if st.checkbox('Relaunch forecast automatically when parameters change', value=True):
    launch_forecast = True
else:
    launch_forecast = st.button('Launch forecast')
if launch_forecast:
    datasets, models, forecasts = forecast_workflow(config, use_cv, make_future_forecast,
                                                    cleaning, params, dates, datasets, models, forecasts)
else:
    st.stop()

# Visualizations

st.write('# 1. Overview')
plot_overview(make_future_forecast, use_cv, models, forecasts)

st.write(f'# 2. Evaluation on {eval["set"].lower()} set')
plot_performance(use_cv, target_col, datasets, forecasts, dates, eval)

st.write('# 3. Impact of components and regressors')
plot_components(use_cv, target_col, models, forecasts, cleaning, resampling)

# st.write('# 4. Future forecast')