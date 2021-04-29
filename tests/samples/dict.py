import pandas as pd
import numpy as np
from lib.inputs.dataprep import _autodetect_dimensions


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
