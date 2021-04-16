import pandas as pd
from loguru import logger


def train_val_split(df: pd.DataFrame,
                    dates: dict
                    ):
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
    return train, val


def make_eval_df(train: pd.DataFrame,
                 val: pd.DataFrame
                 ) -> pd.DataFrame:
    future_eval = pd.DataFrame()
    future_eval['ds'] = train['ds'].to_list() + val['ds'].to_list()
    return future_eval


def make_future_df(df: pd.DataFrame,
                   dates: dict,
                   include_history: bool = True
                   ) -> pd.DataFrame:
    if include_history:
        start_date = df.ds.min()
    else:
        start_date = dates['forecast_start_date']
    future = pd.date_range(start=start_date,
                           end=dates['forecast_end_date'],
                           freq=dates['forecast_freq'][0].upper())
    future = pd.DataFrame(future, columns=['ds'])
    return future
