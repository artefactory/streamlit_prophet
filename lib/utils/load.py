import pandas as pd
from pathlib import Path
import toml
from lib.utils.path import get_project_root

#@st.cache
def load_data(filepath: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    - filepath: str
    Returns
    -------
    pd.DataFrame
        A dataFrame containing the components in columns, for each date (rows)
    """
    try:
        return pd.read_csv(filepath)
    except ValueError as e:
        print(f"{e}, File not found.")
        return None

def initialisation(config_filename):
    config = toml.load(Path(get_project_root()) / f'config/{config_filename}')
    params = dict()
    dates = dict()
    datasets = dict()
    models = dict()
    forecasts = dict()
    return config, params, dates, datasets, models, forecasts
