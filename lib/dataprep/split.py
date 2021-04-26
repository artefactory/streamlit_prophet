import pandas as pd
import streamlit as st
from datetime import timedelta
from lib.dataprep.clean import clean_future_df
from lib.utils.mapping import convert_into_nb_of_days, convert_into_nb_of_seconds


def get_train_val_sets(df: pd.DataFrame, dates: dict) -> dict:
    datasets = dict()
    train = df.query(f'ds >= "{dates["train_start_date"]}" & ds <= "{dates["train_end_date"]}"').copy()
    val = df.query(f'ds >= "{dates["val_start_date"]}" & ds <= "{dates["val_end_date"]}"').copy()
    datasets['train'] = train
    datasets['val'] = val
    datasets['full'] = df.copy()
    st.success(
        f"""Train: {datasets['train'].ds.min().strftime('%Y/%m/%d')} - 
                   {datasets['train'].ds.max().strftime('%Y/%m/%d')}
            Valid: {datasets['val'].ds.min().strftime('%Y/%m/%d')} - 
                   {datasets['val'].ds.max().strftime('%Y/%m/%d')} 
            ({round((len(datasets['val']) / float(len(df)) * 100))}% of data used for validation)""")
    return datasets


def get_train_set(df: pd.DataFrame, dates: dict) -> dict:
    datasets = dict()
    train = df.query(f'ds >= "{dates["train_start_date"]}" & ds <= "{dates["train_end_date"]}"').copy()
    datasets['train'] = train
    datasets['full'] = df.copy()
    return datasets


def make_eval_df(datasets: dict) -> dict:
    eval = pd.concat([datasets['train'], datasets['val']], axis=0)
    eval = eval.drop('y', axis=1)
    datasets['eval'] = eval
    return datasets


def make_future_df(dates: dict, datasets: dict, cleaning: dict, include_history: bool = True) -> dict:
    # TODO: Inclure les valeurs futures de régresseurs ? Pour l'instant, use_regressors = False pour le forecast
    if include_history:
        start_date = datasets['full'].ds.min()
    else:
        start_date = dates['forecast_start_date']
    future = pd.date_range(start=start_date, end=dates['forecast_end_date'], freq=dates['forecast_freq'])
    future = pd.DataFrame(future, columns=['ds'])
    future = clean_future_df(future, cleaning)
    datasets['future'] = future
    return datasets


def get_cv_cutoffs(dates: dict, freq: str) -> list:
    horizon = dates['folds_horizon']
    end = dates['train_end_date']
    n_folds = dates['n_folds']
    if freq in ['s', 'H']:
        multiplier = convert_into_nb_of_seconds(freq, 1)
        cutoffs = [pd.to_datetime(end - timedelta(seconds=(i + 1) * multiplier * horizon)) for i in range(n_folds)]
    else:
        multiplier = convert_into_nb_of_days(freq, 1)
        cutoffs = [pd.to_datetime(end - timedelta(days=(i + 1) * multiplier * horizon)) for i in range(n_folds)]
    return cutoffs


def get_max_possible_cv_horizon(dates: dict, resampling: dict) -> int:
    freq = resampling['freq'][-1]
    if freq in ['s', 'H']:
        divider = convert_into_nb_of_seconds(freq, 1)
        nb_seconds_training = (dates['train_end_date'] - dates['train_start_date']).seconds
        max_horizon = (nb_seconds_training // divider) // dates['n_folds']
    else:
        divider = convert_into_nb_of_days(freq, 1)
        nb_days_training = (dates['train_end_date'] - dates['train_start_date']).days
        max_horizon = (nb_days_training // divider) // dates['n_folds']
    return max_horizon


def prettify_cv_folds_dates(dates: dict, freq: str) -> str:
    # TODO : Remplacer par figure plotly cross-val ?
    horizon = dates['folds_horizon']
    cutoffs_text = []
    for i, cutoff in enumerate(dates['cutoffs']):
        cutoffs_text.append(f"Fold {i + 1}:           ")
        if freq in ['s', 'H']:
            multiplier = convert_into_nb_of_seconds(freq, 1)
            cutoffs_text.append(f"Train: {dates['train_start_date'].strftime('%Y/%m/%d %H:%M:%S')} - "
                                f"{(cutoff - timedelta(seconds=1)).strftime('%Y/%m/%d %H:%M:%S')}")
            cutoffs_text.append(f"Valid: {cutoff.strftime('%Y/%m/%d %H:%M:%S')} - "
                                f"{(cutoff + timedelta(seconds=horizon*multiplier)).strftime('%Y/%m/%d %H:%M:%S')}")
        else:
            multiplier = convert_into_nb_of_days(freq, 1)
            cutoffs_text.append(f"Train: {dates['train_start_date'].strftime('%Y/%m/%d')} - "
                                f"{(cutoff - timedelta(days=1)).strftime('%Y/%m/%d')}")
            cutoffs_text.append(f"Valid: {cutoff.strftime('%Y/%m/%d')} - "
                                f"{(cutoff + timedelta(days=horizon*multiplier)).strftime('%Y/%m/%d')}")
        cutoffs_text.append("")
    return '\n'.join(cutoffs_text)
