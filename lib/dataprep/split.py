import pandas as pd
import streamlit as st
from datetime import timedelta


def get_train_val_sets(df: pd.DataFrame,
                       dates: dict,
                       datasets: dict
                       ) -> dict:
    train = df.query(f'ds >= "{dates["train_start_date"]}" & ds <= "{dates["train_end_date"]}"').copy()
    val = df.query(f'ds >= "{dates["val_start_date"]}" & ds <= "{dates["val_end_date"]}"').copy()
    datasets['train'] = train
    datasets['val'] = val
    datasets['full'] = df.copy()
    st.success(
        f"""Train: {datasets['train'].ds.min().strftime('%d/%m/%Y')} - 
                   {datasets['train'].ds.max().strftime('%d/%m/%Y')}
            Valid: {datasets['val'].ds.min().strftime('%d/%m/%Y')} - 
                   {datasets['val'].ds.max().strftime('%d/%m/%Y')} 
            ({round((len(datasets['val']) / float(len(df)) * 100))}% of data used for validation)""")
    return datasets


def get_train_set(df: pd.DataFrame,
                  dates: dict,
                  datasets: dict
                  ) -> dict:
    train = df.query(f'ds >= "{dates["train_start_date"]}" & ds <= "{dates["train_end_date"]}"').copy()
    datasets['train'] = train
    datasets['full'] = df.copy()
    return datasets


def make_eval_df(datasets: dict) -> dict:
    eval = pd.concat([datasets['train'], datasets['val']], axis=0)
    eval = eval.drop('y', axis=1)
    datasets['eval'] = eval
    return datasets


def make_future_df(dates: dict, datasets: dict, include_history: bool = True) -> dict:
    # TODO: Inclure les valeurs futures de régresseurs ? Pour l'instant, use_regressors = False pour le forecast
    if include_history:
        start_date = datasets['full'].ds.min()
    else:
        start_date = dates['forecast_start_date']
    future = pd.date_range(start=start_date,
                           end=dates['forecast_end_date'],
                           freq=dates['forecast_freq'][0].upper())
    future = pd.DataFrame(future, columns=['ds'])
    datasets['future'] = future
    return datasets


def get_cv_cutoffs(dates: dict) -> list:
    cutoffs = [pd.to_datetime(dates['train_end_date'] - timedelta(days=(i + 1) * dates['folds_horizon'])) \
               for i in range(dates['n_folds'])]
    return cutoffs


def get_max_possible_cv_horizon(dates: dict) -> int:
    nb_days_training = (dates['train_end_date'] - dates['train_start_date']).days
    return nb_days_training // dates['n_folds']


def prettify_cv_folds_dates(dates):
    # TODO : Remplacer par figure plotly cross-val ?
    cutoffs_text = []
    for i, cutoff in enumerate(dates['cutoffs']):
        cutoffs_text.append(f"Fold {i + 1}:           ")
        cutoffs_text.append(f"Train: {dates['train_start_date'].strftime('%d-%m-%Y')} - "
                            f"{(cutoff - timedelta(days=1)).strftime('%d-%m-%Y')}")
        cutoffs_text.append(f"Valid: {cutoff.strftime('%d-%m-%Y')} - "
                            f"{(cutoff + timedelta(days=dates['folds_horizon'])).strftime('%d-%m-%Y')}")
        cutoffs_text.append("")
    return '\n'.join(cutoffs_text)
