import pandas as pd
from datetime import timedelta
from loguru import logger
from typing import MutableMapping, Optional, Union

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
        #TODO: raise TrainingError
        logger.warning('Training set is empty')
    if len(val) == 0:
        logger.warning("There is no validation set. Evaluation won't be possible.")
    return train, val

def make_future_eval_df(train: pd.DataFrame,
                        val: pd.DataFrame
                        ) -> pd.DataFrame:
    future = pd.DataFrame()
    future['ds'] = train['ds'].to_list() + val['ds'].to_list()
    return future