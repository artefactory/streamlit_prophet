import streamlit as st
from datetime import timedelta
from lib.utils.mapping import convert_into_nb_of_days
from lib.dataprep.split import get_max_possible_cv_horizon, get_cv_cutoffs, prettify_cv_folds_dates


def input_train_dates(df, dates, use_cv):
    set_name = "CV" if use_cv else "Training"
    dates['train_start_date'] = st.date_input(
        f"{set_name} start date",
        value=df.ds.min(),
        min_value=df.ds.min(),
        max_value=df.ds.max(),
    )
    dates['train_end_date'] = st.date_input(
        f"{set_name} end date",
        value=df.ds.max() - timedelta(days=30), # TODO: Gérer le edge case où le dataset fait moins de 30 jours
        min_value=dates['train_start_date'] + timedelta(days=1),
        max_value=df.ds.max(),
    )
    return dates


def input_val_dates(df, dates):
    dates['val_start_date'] = st.date_input(
        "Validation start date",
        value=dates['train_end_date'] + timedelta(days=1), # TODO: Gérer le edge case où la train end date entrée est df.ds.max()
        min_value=dates['train_end_date'] + timedelta(days=1),
        max_value=df.ds.max(),
    )
    dates['val_end_date'] = st.date_input(
        "Validation end date",
        value=df.ds.max(),
        min_value=dates['val_start_date'] + timedelta(days=1),
        max_value=df.ds.max(),
    )
    return dates


def input_cv(dates):
    dates['n_folds'] = st.number_input("Number of CV folds", min_value=1, value=5)
    max_possible_horizon = get_max_possible_cv_horizon(dates)
    dates['folds_horizon'] = st.number_input("Horizon of each fold (in days)",
                                             min_value=1,
                                             max_value=max_possible_horizon,
                                             value=min(30,max_possible_horizon)
                                             )
    dates['cutoffs'] = get_cv_cutoffs(dates)
    st.success(prettify_cv_folds_dates(dates))
    return dates


def input_forecast_dates(df, dates, config):
    forecast_freq = st.selectbox("Granularity of prediction", config["forecast"]["freq"])
    forecast_horizon = st.number_input(f"Forecast horizon in {forecast_freq}s",
                                       min_value=1, value=10)
    right_after = st.checkbox("Start forecasting right after the most recent date in dataset", value=True)
    if right_after:
        dates['forecast_start_date'] = df.ds.max() + timedelta(days=1)
    else:
        dates['forecast_start_date'] = st.date_input(
            "Forecast start date:",
            value=df.ds.max(),
            min_value=df.ds.max(),
        )
    timedelta_horizon = convert_into_nb_of_days(forecast_freq, forecast_horizon)
    dates['forecast_freq'] = forecast_freq
    dates['forecast_end_date'] = dates['forecast_start_date'] + timedelta(days=timedelta_horizon)
    st.success(
        f"""Forecast: {dates['forecast_start_date'].strftime('%d/%m/%Y')} - 
                      {dates['forecast_end_date'].strftime('%d/%m/%Y')}""")
    return dates
