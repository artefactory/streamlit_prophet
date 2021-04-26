import streamlit as st
import pandas as pd
from datetime import timedelta
from lib.utils.mapping import convert_into_nb_of_days, convert_into_nb_of_seconds, mapping_freq_names
from lib.dataprep.split import get_max_possible_cv_horizon, get_cv_cutoffs, prettify_cv_folds_dates


def input_train_dates(df: pd.DataFrame, use_cv: bool) -> dict:
    # TODO: Gérer le edge case où le dataset fait moins de 30 jours
    # TODO : Choisir les valeurs par défaut en fonction de resampling
    dates = dict()
    set_name = "CV" if use_cv else "Training"
    dates['train_start_date'] = st.date_input(f"{set_name} start date",
                                              value=df.ds.min(),
                                              min_value=df.ds.min(),
                                              max_value=df.ds.max()
                                              )
    dates['train_end_date'] = st.date_input(f"{set_name} end date",
                                            value=df.ds.max() if use_cv else df.ds.max() - timedelta(days=30),
                                            min_value=dates['train_start_date'] + timedelta(days=1),
                                            max_value=df.ds.max()
                                            )
    return dates


def input_val_dates(df: pd.DataFrame, dates: dict) -> dict:
    # TODO: Gérer le edge case où la train end date entrée est df.ds.max()
    # TODO : Choisir les valeurs par défaut en fonction de resampling
    dates['val_start_date'] = st.date_input("Validation start date",
                                            value=dates['train_end_date'] + timedelta(days=1),
                                            min_value=dates['train_end_date'] + timedelta(days=1),
                                            max_value=df.ds.max()
                                            )
    dates['val_end_date'] = st.date_input("Validation end date",
                                          value=df.ds.max(),
                                          min_value=dates['val_start_date'] + timedelta(days=1),
                                          max_value=df.ds.max()
                                          )
    return dates


def input_cv(dates: dict, resampling: dict) -> dict:
    dates['n_folds'] = st.number_input("Number of CV folds", min_value=1, value=5)
    freq = resampling['freq'][-1]
    freq_name = mapping_freq_names(freq)
    max_possible_horizon = get_max_possible_cv_horizon(dates, resampling)
    default_values = {'H': 96, 'D': 30, 'W': 10, 'M': 6, 'Y': 3}
    dates['folds_horizon'] = st.number_input(f"Horizon of each fold (in {freq_name})",
                                             min_value=1,
                                             max_value=max_possible_horizon,
                                             value=min(default_values[freq], max_possible_horizon)
                                             )
    dates['cutoffs'] = get_cv_cutoffs(dates, freq)
    st.success(prettify_cv_folds_dates(dates, freq))
    return dates


def input_forecast_dates(df: pd.DataFrame, dates: dict, resampling: dict) -> dict:
    # TODO : Tester avec un dataset à la granularité heure et minute
    forecast_freq_name = mapping_freq_names(resampling['freq'][-1])
    forecast_horizon = st.number_input(f"Forecast horizon in {forecast_freq_name}", min_value=1, value=10)
    right_after = st.checkbox("Start forecasting right after the most recent date in dataset", value=True)
    if right_after:
        if forecast_freq_name in ['seconds', 'hours']:
            dates['forecast_start_date'] = df.ds.max() + timedelta(seconds=1)
        else:
            dates['forecast_start_date'] = df.ds.max() + timedelta(days=1)
    else:
        dates['forecast_start_date'] = st.date_input("Forecast start date:", value=df.ds.max(), min_value=df.ds.max())
    if forecast_freq_name in ['seconds', 'hours']:
        timedelta_horizon = convert_into_nb_of_seconds(resampling['freq'][-1], forecast_horizon)
        dates['forecast_end_date'] = dates['forecast_start_date'] + timedelta(seconds=timedelta_horizon)
        st.success(f"""Forecast: {dates['forecast_start_date'].strftime('%Y/%m/%d %H:%M:%S')} - 
                                 {dates['forecast_end_date'].strftime('%Y/%m/%d %H:%M:%S')}""")
    else:
        timedelta_horizon = convert_into_nb_of_days(resampling['freq'][-1], forecast_horizon)
        dates['forecast_end_date'] = dates['forecast_start_date'] + timedelta(days=timedelta_horizon)
        st.success(f"""Forecast: {dates['forecast_start_date'].strftime('%Y/%m/%d')} - 
                                 {dates['forecast_end_date'].strftime('%Y/%m/%d')}""")
    dates['forecast_freq'] = str(resampling['freq'])
    return dates
