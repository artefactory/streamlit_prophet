import pandas as pd
import numpy as np
from lib.utils.load import load_config, download_toy_dataset

config, _ = load_config('config_streamlit.toml', 'config_readme.toml')


def make_df(ds=None, cols=None, start='2010-01-01', end='2015-01-01', freq='D'):
    df = pd.DataFrame()
    if ds is not None:
        df['ds'] = pd.date_range(start=start if 'start_date' not in ds.keys() else ds['start_date'],
                                 end=end if 'end_date' not in ds.keys() else ds['end_date'],
                                 freq=freq if 'freq' not in ds.keys() else ds['freq'])
        if 'str' in ds.keys():
            df['ds'] = df['ds'].map(lambda x: x.strftime(ds['str']))
        if 'frac_nan' in ds.keys():
            df.loc[df.sample(frac=ds['frac_nan']).index, 'ds'] = np.nan
    if cols is not None:
        N = len(df) if len(df) > 0 else 100
        for col in cols.keys():
            if 'cat' in cols[col].keys():
                df[col] = np.random.choice(a=cols[col]['cat'], size=N)
            elif 'range' in cols[col].keys():
                df[col] = np.random.randn(1, N).ravel() * cols[col]['range']
                if 'abs' in cols[col].keys():
                    df[col] = abs(df[col])
            if 'frac_nan' in cols[col].keys():
                df.loc[df.sample(frac=cols[col]['frac_nan']).index, col] = np.nan
    return df


dataframes = dict()
dataframes[0] = pd.DataFrame()
dataframes[1] = make_df(cols={0: {'cat': ['A', 'B', 'C']},
                              1: {'cat': ['A', 'B']},
                              2: {'cat': ['A']}
                              }
                        )
dataframes[2] = make_df(cols={0: {'cat': ['A'], 'frac_nan': 1},
                              1: {'cat': ['A'], 'frac_nan': 0.5},
                              2: {'cat': ['A']}
                              }
                        )
dataframes[3] = make_df(ds={})
dataframes[4] = make_df(ds={'str': '%Y-%m-%d'})
dataframes[5] = make_df(ds={'freq': 'Y'})
dataframes[6] = make_df(ds={'freq': 'H'})
dataframes[7] = make_df(ds={'frac_nan': 0.5})

for dataset in config['datasets'].keys():
    dataframes[dataset] = download_toy_dataset(config['datasets'][dataset]['url'])
