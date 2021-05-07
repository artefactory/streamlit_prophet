from typing import Tuple

import io
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import toml


def get_project_root() -> str:
    """Returns project root path.

    Returns
    -------
    str
        Project root path.
    """
    return str(Path(__file__).parent.parent.parent)


@st.cache()
def load_dataset(file) -> pd.DataFrame:
    """Loads dataset from user's file system as a pandas dataframe.

    Parameters
    ----------
    file : str
        Dataset file path.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    try:
        return pd.read_csv(file)
    except:
        st.error(
            "This file can't be converted into a dataframe. Please import a csv file with ',' as a separator."
        )
        st.stop()


@st.cache()
def load_config(config_streamlit_filename: str, config_readme_filename: str) -> Tuple[dict, dict]:
    """Loads configuration files.

    Parameters
    ----------
    config_streamlit_filename : str
        Filename of lib configuration file.
    config_readme_filename : str
        Filename of readme configuration file.

    Returns
    -------
    dict
        Lib configuration file.
    dict
        Readme configuration file.
    """
    config_streamlit = toml.load(Path(get_project_root()) / f"config/{config_streamlit_filename}")
    config_readme = toml.load(Path(get_project_root()) / f"config/{config_readme_filename}")
    return config_streamlit, config_readme


@st.cache()
def download_toy_dataset(url: str) -> pd.DataFrame:
    """Downloads a toy dataset from an external source and converts it into a pandas dataframe.

    Parameters
    ----------
    url : str
        Link to the toy dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    download = requests.get(url).content
    df = pd.read_csv(io.StringIO(download.decode("utf-8")))
    return df
