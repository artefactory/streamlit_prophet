import streamlit as st
from datetime import timedelta

def input_split_dates(df, dates):
    dates['train_start_date'] = st.date_input(
        "Training start date:",
        value=df.ds.min(),
        min_value=df.ds.min(),
        max_value=df.ds.max(),
    )
    dates['val_start_date'] = st.date_input(
        "Validation start date:",
        value=df.ds.max() - timedelta(days=30),
        min_value=dates['train_start_date'],
        max_value=df.ds.max(),
    )
    dates['val_end_date'] = st.date_input(
        "Validation end date:",
        value=df.ds.max(),
        min_value=dates['val_start_date'],
        max_value=df.ds.max(),
    )
    return dates

def input_cv_dates(df, dates):
    dates['cv_start_date'] = st.date_input(
        "CV start date:",
        value=df.ds.min(),
        min_value=df.ds.min(),
        max_value=df.ds.max(),
    )
    dates['cv_end_date'] = st.date_input(
        "CV end date:",
        value=df.ds.max(),
        min_value=dates['cv_start_date'],
        max_value=df.ds.max(),
    )
    dates['n_folds'] = st.number_input("Number of CV folds:", min_value=1, value=5)
    return dates

def input_forecast_dates(df, dates):
    forecast_horizon = st.number_input(
        "Forecast horizon in days", min_value=1, value=round(len(df) * 0.2))
    right_after = st.checkbox("Start forecasting right after the most recent date in dataset", value=True)
    if right_after:
        dates['forecast_start_date'] = df.ds.max() + timedelta(days=1)
    else:
        dates['forecast_start_date'] = st.date_input(
            "Forecast start date:",
            value=df.ds.max(),
            min_value=df.ds.max(),
        )
    dates['forecast_end_date'] = dates['forecast_start_date'] + timedelta(days=forecast_horizon)
    return dates