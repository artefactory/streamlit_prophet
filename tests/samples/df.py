from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from streamlit_prophet.lib.utils.load import load_config

config, _, _ = load_config(
    "config_streamlit.toml", "config_instructions.toml", "config_readme.toml"
)


def make_test_df(
    ds: Optional[Dict[Any, Any]] = None,
    cols: Optional[Dict[Any, Any]] = None,
    start: str = "2010-01-01",
    end: str = "2020-01-01",
    freq: str = "D",
    range: int = 10,
) -> pd.DataFrame:
    """Creates a sample dataframe with specifications defined by the arguments, for testing purpose.

    Parameters
    ----------
    ds : Optional[dict]
        Specifications for date column.
    cols : Optional[dict]
        Specifications for other columns.
    start : str
        Start date for date column.
    end : str
        End date for date column.
    freq : str
        Frequency for date column.
    range : int
        Range for numerical columns.

    Returns
    -------
    pd.DataFrame
        Dataframe that will be used for unit tests.
    """
    df = pd.DataFrame()
    if ds is not None:
        df["ds"] = pd.date_range(
            start=start if "start_date" not in ds.keys() else ds["start_date"],
            end=end if "end_date" not in ds.keys() else ds["end_date"],
            freq=freq if "freq" not in ds.keys() else ds["freq"],
        )
        if "str" in ds.keys():
            df["ds"] = df["ds"].map(lambda x: x.strftime(ds["str"]))
        if "frac_nan" in ds.keys():
            df.loc[df.sample(frac=ds["frac_nan"]).index, "ds"] = np.nan
    if cols is not None:
        N = len(df) if len(df) > 0 else 1000
        for col in cols.keys():
            if "cat" in cols[col].keys():
                df[col] = np.random.choice(a=cols[col]["cat"], size=N)
            else:
                range = range if "range" not in cols[col].keys() else cols[col]["range"]
                df[col] = np.random.randn(1, N).ravel() * range
                if "abs" in cols[col].keys():
                    df[col] = abs(df[col])
            if "frac_nan" in cols[col].keys():
                df.loc[df.sample(frac=cols[col]["frac_nan"]).index, col] = np.nan
    return df


# Synthetic categorical variables
int_long_target = list(range(1, config["validity"]["min_target_cardinality"] + 2))
int_short_target = list(range(1, config["validity"]["min_target_cardinality"] - 1))
int_long_cat = list(range(1, config["validity"]["max_cat_reg_cardinality"] + 2))
int_short_cat = list(range(1, config["validity"]["max_cat_reg_cardinality"] - 1))
str_long_target = [
    chr(ord("@") + i) for i in range(1, config["validity"]["min_target_cardinality"] + 2)
]
str_short_target = [
    chr(ord("@") + i) for i in range(1, config["validity"]["min_target_cardinality"] - 1)
]
str_long_cat = [
    chr(ord("@") + i) for i in range(1, config["validity"]["max_cat_reg_cardinality"] + 2)
]
str_short_cat = [
    chr(ord("@") + i) for i in range(1, config["validity"]["max_cat_reg_cardinality"] - 1)
]

# Test dataframes
df_test = dict()
df_test[0] = pd.DataFrame()
df_test[1] = make_test_df(
    cols={0: {"cat": ["A", "B", "C"]}, 1: {"cat": ["A", "B"]}, 2: {"cat": ["A"]}, 3: {}}
)
df_test[2] = make_test_df(
    cols={0: {"cat": ["A"], "frac_nan": 1}, 1: {"cat": ["A"], "frac_nan": 0.1}, 2: {"cat": ["A"]}}
)
df_test[3] = make_test_df(cols={"y": {"cat": int_short_target}})
df_test[4] = make_test_df(cols={"y": {"cat": int_short_target, "frac_nan": 0.1}})
df_test[5] = make_test_df(cols={"y": {"cat": int_short_target, "frac_nan": 1}})
df_test[6] = make_test_df(cols={"y": {"cat": str_long_target}})
df_test[7] = make_test_df(cols={"y": {"cat": str_long_target, "frac_nan": 0.1}})
df_test[8] = make_test_df(ds={}, cols={"y": {"cat": int_long_target}})
df_test[9] = make_test_df(
    ds={"str": "%Y-%m-%d"}, cols={"y": {"cat": int_long_target, "frac_nan": 0.1}}
)
df_test[10] = make_test_df(ds={"freq": "Y"}, cols={"y": {"range": 100}})
df_test[11] = make_test_df(ds={"freq": "H"}, cols={"y": {"range": 1, "abs": True}})
df_test[12] = make_test_df(ds={"frac_nan": 0.1}, cols={"y": {"range": 1, "frac_nan": 0.1}})
df_test[13] = make_test_df(
    cols={
        0: {},
        1: {"frac_nan": 0.1},
        2: {"frac_nan": 1},
        3: {"abs": True},
        4: {"cat": int_short_cat},
        5: {"cat": int_short_cat, "frac_nan": 0.1},
        6: {"cat": str_short_cat, "frac_nan": 0.1},
    }
)
df_test[14] = lambda x: make_test_df(
    ds={"freq": x},
    cols={
        "y": {},
        0: {},
        1: {"frac_nan": 0.1},
        2: {"frac_nan": 1},
        3: {"abs": True},
        4: {"cat": int_short_cat},
        5: {"cat": int_short_cat, "frac_nan": 0.1},
        6: {"cat": str_short_cat, "frac_nan": 0.1},
        7: {"cat": str_long_cat},
        8: {"cat": int_long_cat},
        9: {"cat": str_long_cat, "frac_nan": 0.1},
        10: {"cat": int_long_cat, "frac_nan": 0.1},
        11: {"cat": ["A"]},
        12: {"cat": ["A"], "frac_nan": 0.1},
    },
)
df_test[15] = make_test_df(cols={"y": {"cat": [2]}})
df_test[16] = make_test_df(cols={"y": {"cat": [3]}})
df_test[17] = make_test_df(ds={}, cols={"truth": {}, "forecast": {}})
df_test[18] = make_test_df(ds={}, cols={"truth": {"frac_nan": 1}, "forecast": {"frac_nan": 1}})
df_test[19] = make_test_df(ds={"freq": "W"}, cols={"truth": {"frac_nan": 0.1}, "forecast": {}})
df_test[20] = make_test_df(ds={}, cols={"y": {}, "regressor1": {}, "regressor2": {"cat": [0, 1]}})
