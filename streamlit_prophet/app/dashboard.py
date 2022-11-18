from typing import Any, Dict, List

import streamlit as st
from streamlit_prophet.lib.dataprep.clean import clean_df
from streamlit_prophet.lib.dataprep.format import (
    add_cap_and_floor_cols,
    check_dataset_size,
    filter_and_aggregate_df,
    format_date_and_target,
    format_datetime,
    print_empty_cols,
    print_removed_cols,
    remove_empty_cols,
    resample_df,
)
from streamlit_prophet.lib.dataprep.split import get_train_set, get_train_val_sets
from streamlit_prophet.lib.exposition.export import display_links, display_save_experiment_button
from streamlit_prophet.lib.exposition.visualize import (
    plot_components,
    plot_future,
    plot_overview,
    plot_performance,
)
from streamlit_prophet.lib.inputs.dataprep import input_cleaning, input_dimensions, input_resampling
from streamlit_prophet.lib.inputs.dataset import (
    input_columns,
    input_dataset,
    input_future_regressors,
)
from streamlit_prophet.lib.inputs.dates import (
    input_cv,
    input_forecast_dates,
    input_train_dates,
    input_val_dates,
)
from streamlit_prophet.lib.inputs.eval import input_metrics, input_scope_eval
from streamlit_prophet.lib.inputs.params import (
    input_holidays_params,
    input_other_params,
    input_prior_scale_params,
    input_regressors,
    input_seasonality_params,
)
from streamlit_prophet.lib.models.prophet import forecast_workflow
from streamlit_prophet.lib.utils.load import load_config, load_image

# Page config
st.set_page_config(page_title="Prophet", layout="wide")

# Load config
config, instructions, readme = load_config(
    "config_streamlit.toml", "config_instructions.toml", "config_readme.toml"
)

# Initialization
dates: Dict[Any, Any] = dict()
report: List[Dict[str, Any]] = []

# Info
with st.expander(
    "Streamlit app to build a time series forecasting model in a few clicks", expanded=False
):
    st.write(readme["app"]["app_intro"])
    st.write("")
st.write("")
st.sidebar.image(load_image("logo.png"), use_column_width=True)
display_links(readme["links"]["repo"], readme["links"]["article"])


st.sidebar.title("1. Data")

# Load data
with st.sidebar.expander("Dataset", expanded=True):
    df, load_options, config, datasets = input_dataset(config, readme, instructions)
    df, empty_cols = remove_empty_cols(df)
    print_empty_cols(empty_cols)

# Column names
with st.sidebar.expander("Columns", expanded=True):
    date_col, target_col = input_columns(config, readme, df, load_options)
    df = format_date_and_target(df, date_col, target_col, config, load_options)

# Filtering
with st.sidebar.expander("Filtering", expanded=False):
    dimensions = input_dimensions(df, readme, config)
    df, cols_to_drop = filter_and_aggregate_df(df, dimensions, config, date_col, target_col)
    print_removed_cols(cols_to_drop)

# Resampling
with st.sidebar.expander("Resampling", expanded=False):
    resampling = input_resampling(df, readme)
    df = format_datetime(df, resampling)
    df = resample_df(df, resampling)
    check_dataset_size(df, config)

# Cleaning
with st.sidebar.expander("Cleaning", expanded=False):
    cleaning = input_cleaning(resampling, readme, config)
    df = clean_df(df, cleaning)
    check_dataset_size(df, config)

st.sidebar.title("2. Modelling")

# Prior scale
with st.sidebar.expander("Prior scale", expanded=False):
    params = input_prior_scale_params(config, readme)

# Seasonalities
with st.sidebar.expander("Seasonalities", expanded=False):
    params = input_seasonality_params(config, params, resampling, readme)

# Holidays
with st.sidebar.expander("Holidays"):
    params = input_holidays_params(params, readme, config)

# External regressors
with st.sidebar.expander("Regressors"):
    params = input_regressors(df, config, params, readme)

# Other parameters
with st.sidebar.expander("Other parameters", expanded=False):
    params = input_other_params(config, params, readme)
    df = add_cap_and_floor_cols(df, params)

st.sidebar.title("3. Evaluation")

# Choose whether or not to do evaluation
evaluate = st.sidebar.checkbox(
    "Evaluate my model", value=True, help=readme["tooltips"]["choice_eval"]
)

if evaluate:

    # Split
    with st.sidebar.expander("Split", expanded=True):
        use_cv = st.checkbox(
            "Perform cross-validation", value=False, help=readme["tooltips"]["choice_cv"]
        )
        dates = input_train_dates(df, use_cv, config, resampling, dates)
        if use_cv:
            dates = input_cv(dates, resampling, config, readme)
            datasets = get_train_set(df, dates, datasets)
        else:
            dates = input_val_dates(df, dates, config)
            datasets = get_train_val_sets(df, dates, config, datasets)

    # Performance metrics
    with st.sidebar.expander("Metrics", expanded=False):
        eval = input_metrics(readme, config)

    # Scope of evaluation
    with st.sidebar.expander("Scope", expanded=False):
        eval = input_scope_eval(eval, use_cv, readme)

else:
    use_cv = False

st.sidebar.title("4. Forecast")

# Choose whether or not to do future forecasts
make_future_forecast = st.sidebar.checkbox(
    "Make forecast on future dates", value=False, help=readme["tooltips"]["choice_forecast"]
)
if make_future_forecast:
    with st.sidebar.expander("Horizon", expanded=False):
        dates = input_forecast_dates(df, dates, resampling, config, readme)
    with st.sidebar.expander("Regressors", expanded=False):
        datasets = input_future_regressors(
            datasets, dates, params, dimensions, load_options, date_col
        )

# Launch training & forecast
if st.checkbox(
    "Launch forecast",
    value=False,
    help=readme["tooltips"]["launch_forecast"],
):

    if not (evaluate | make_future_forecast):
        st.error("Please check at least 'Evaluation' or 'Forecast' in the sidebar.")

    track_experiments = st.checkbox(
        "Track experiments", value=False, help=readme["tooltips"]["track_experiments"]
    )

    datasets, models, forecasts = forecast_workflow(
        config,
        use_cv,
        make_future_forecast,
        evaluate,
        cleaning,
        resampling,
        params,
        dates,
        datasets,
        df,
        date_col,
        target_col,
        dimensions,
        load_options,
    )

    # Visualizations

    if evaluate | make_future_forecast:
        st.write("# 1. Overview")
        report = plot_overview(
            make_future_forecast, use_cv, models, forecasts, target_col, cleaning, readme, report
        )

    if evaluate:
        st.write(
            f'# 2. Evaluation on {"CV" if use_cv else ""} {eval["set"].lower()} set{"s" if use_cv else ""}'
        )
        report = plot_performance(
            use_cv, target_col, datasets, forecasts, dates, eval, resampling, config, readme, report
        )

    if evaluate | make_future_forecast:
        st.write(
            "# 3. Impact of components and regressors"
            if evaluate
            else "# 2. Impact of components and regressors"
        )
        report = plot_components(
            use_cv,
            make_future_forecast,
            target_col,
            models,
            forecasts,
            cleaning,
            resampling,
            config,
            readme,
            df,
            report,
        )

    if make_future_forecast:
        st.write("# 4. Future forecast" if evaluate else "# 3. Future forecast")
        report = plot_future(models, forecasts, dates, target_col, cleaning, readme, report)

    # Save experiment
    if track_experiments:
        display_save_experiment_button(
            report,
            config,
            use_cv,
            make_future_forecast,
            evaluate,
            cleaning,
            resampling,
            params,
            dates,
            date_col,
            target_col,
            dimensions,
        )
