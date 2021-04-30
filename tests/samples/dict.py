import pandas as pd
import numpy as np
from datetime import datetime
from lib.inputs.dataprep import _autodetect_dimensions
from lib.dataprep.split import get_cv_cutoffs
from lib.utils.load import load_config

config, _ = load_config('config_streamlit.toml', 'config_readme.toml')


# Resampling
def make_resampling_test(freq='D', resample=True, agg='Mean') -> dict:
    return {'freq': freq, 'resample': resample, 'agg': agg}


# Dimensions
def make_dimensions_test(df: pd.DataFrame, frac=0.5, agg='Mean') -> dict:
    dimensions = dict()
    dimensions_cols = _autodetect_dimensions(df)
    for col in dimensions_cols:
        values = list(df[col].unique())
        dimensions[col] = np.random.choice(values, size=int(len(values) * frac), replace=False)
    if len(dimensions_cols) > 0:
        dimensions['agg'] = agg
    return dimensions


# Cleaning
def make_cleaning_test(del_days=[], del_zeros=True, del_negative=True, log_transform=False) -> dict:
    return {'del_days': del_days, 'del_zeros': del_zeros, 'del_negative': del_negative, 'log_transform': log_transform}


# Dates
def make_dates_test(train_start='2010-01-01',
                    train_end='2014-12-31',
                    val_start='2015-01-01',
                    val_end='2019-12-31',
                    forecast_start='2020-01-01',
                    forecast_end='2020-12-31',
                    n_folds=5,
                    folds_horizon=0,
                    freq='D',
                    ):
    dates = {'train_start_date': datetime.date(datetime.strptime(train_start, '%Y-%m-%d')),
             'train_end_date': datetime.date(datetime.strptime(train_end, '%Y-%m-%d')),
             'val_start': datetime.date(datetime.strptime(val_start, '%Y-%m-%d')),
             'val_end': datetime.date(datetime.strptime(val_end, '%Y-%m-%d')),
             'forecast_start_date': datetime.date(datetime.strptime(forecast_start, '%Y-%m-%d')),
             'forecast_end_date': datetime.date(datetime.strptime(forecast_end, '%Y-%m-%d')),
             'forecast_freq': freq,
             'n_folds': n_folds,
             'freq': freq,
             'folds_horizon': config['horizon'][freq[-1]] if folds_horizon == 0 else folds_horizon}
    dates['cutoffs'] = get_cv_cutoffs(dates, freq[-1])
    return dates
