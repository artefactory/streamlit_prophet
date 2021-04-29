import pandas as pd
import numpy as np
from lib.utils.load import load_config, download_toy_dataset

config, _ = load_config('config_streamlit.toml', 'config_readme.toml')


def make_test_df(ds=None, cols=None, start='2010-01-01', end='2020-01-01', freq='D', range=10):
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
        N = len(df) if len(df) > 0 else 1000
        for col in cols.keys():
            if 'cat' in cols[col].keys():
                df[col] = np.random.choice(a=cols[col]['cat'], size=N)
            else:
                range = range if 'range' not in cols[col].keys() else cols[col]['range']
                df[col] = np.random.randn(1, N).ravel() * range
                if 'abs' in cols[col].keys():
                    df[col] = abs(df[col])
            if 'frac_nan' in cols[col].keys():
                df.loc[df.sample(frac=cols[col]['frac_nan']).index, col] = np.nan
    return df


df_test = dict()

# Synthetic dataframes
df_test[0] = pd.DataFrame()
df_test[1] = make_test_df(cols={0: {'cat': ['A', 'B', 'C']},
                                1: {'cat': ['A', 'B']},
                                2: {'cat': ['A']}})
df_test[2] = make_test_df(cols={0: {'cat': ['A'], 'frac_nan': 1},
                                1: {'cat': ['A'], 'frac_nan': 0.1},
                                2: {'cat': ['A']}})
df_test[3] = make_test_df(cols={'y': {'cat': [1, 2, 3]}})
df_test[4] = make_test_df(cols={'y': {'cat': [1, 2, 3], 'frac_nan': 0.1}})
df_test[5] = make_test_df(cols={'y': {'cat': [1, 2, 3], 'frac_nan': 1}})
df_test[6] = make_test_df(cols={'y': {'cat': ['A', 'B', 'C', 'D', 'E', 'F']}})
df_test[7] = make_test_df(cols={'y': {'cat': ['A', 'B', 'C', 'D', 'E', 'F'], 'frac_nan': 0.1}})
df_test[8] = make_test_df(ds={},
                          cols={'y': {'cat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}})
df_test[9] = make_test_df(ds={'str': '%Y-%m-%d'},
                          cols={'y': {'cat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'frac_nan': 0.1}})
df_test[10] = make_test_df(ds={'freq': 'Y'},
                           cols={'y': {'range': 100}})
df_test[11] = make_test_df(ds={'freq': 'H'},
                           cols={'y': {'range': 1, 'abs': True}})
df_test[12] = make_test_df(ds={'frac_nan': 0.1},
                           cols={'y': {'range': 1, 'frac_nan': 0.1}})
df_test[13] = make_test_df(cols={0: {},
                                 1: {'frac_nan': 0.1},
                                 2: {'frac_nan': 1},
                                 3: {'abs': True},
                                 4: {'cat': [1, 2, 3]},
                                 5: {'cat': [1, 2, 3], 'frac_nan': 0.1},
                                 6: {'cat': ['A', 'B', 'C'], 'frac_nan': 0.1}})
df_test[14] = lambda x: make_test_df(ds={'freq': x},
                                     cols={'y': {},
                                           0: {},
                                           1: {'frac_nan': 0.1},
                                           2: {'frac_nan': 1},
                                           3: {'abs': True},
                                           4: {'cat': [1, 2, 3]},
                                           5: {'cat': [1, 2, 3], 'frac_nan': 0.1},
                                           6: {'cat': ['A', 'B', 'C'], 'frac_nan': 0.1},
                                           7: {'cat': ['A', 'B', 'C', 'D', 'E', 'F']},
                                           8: {'cat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                                           9: {'cat': ['A', 'B', 'C', 'D', 'E', 'F'], 'frac_nan': 0.1},
                                           10: {'cat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'frac_nan': 0.1},
                                           11: {'cat': ['A']},
                                           12: {'cat': ['A'], 'frac_nan': 0.1}})

# Toy dataframes
for dataset in config['datasets'].keys():
    df_test[dataset] = download_toy_dataset(config['datasets'][dataset]['url'])
