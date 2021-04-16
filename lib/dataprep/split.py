import pandas as pd
from loguru import logger


def train_val_split(df: pd.DataFrame,
                    dates: dict,
                    datasets: dict
                    ) -> dict:
    """Split dataset according to a split date in format "YYYY-MM-DD"
    - train: [:split_date[
    - test: [split_date:]
    """
    train = df.query(f'ds >= "{dates["train_start_date"]}" & ds < "{dates["val_start_date"]}"').copy()
    val = df.query(f'ds >= "{dates["val_start_date"]}" & ds <= "{dates["val_end_date"]}"').copy()
    if len(train) == 0:
        # TODO: raise TrainingError
        logger.warning('Training set is empty')
    if len(val) == 0:
        logger.warning("There is no validation set. Evaluation won't be possible.")
    datasets['train'] = train
    datasets['val'] = val
    return datasets


def make_eval_df(datasets: dict) -> dict:
    eval = pd.DataFrame()
    eval['ds'] = datasets['train']['ds'].to_list() + datasets['val']['ds'].to_list()
    datasets['eval'] = eval
    return datasets


def make_future_df(df: pd.DataFrame,
                   dates: dict,
                   datasets: dict,
                   include_history: bool = True
                   ) -> dict:
    if include_history:
        start_date = df.ds.min()
    else:
        start_date = dates['forecast_start_date']
    future = pd.date_range(start=start_date,
                           end=dates['forecast_end_date'],
                           freq=dates['forecast_freq'][0].upper())
    future = pd.DataFrame(future, columns=['ds'])
    datasets['future'] = future
    return datasets
