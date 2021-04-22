import pandas as pd
from pathlib import Path
import toml
from lib.utils.path import get_project_root
import streamlit as st
import requests
import io


@st.cache()
def load_dataset(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file)
    except ValueError as e:
        print(f"{e}, File not found.")
        return None


@st.cache()
def load_config(config_streamlit_filename: str,
                config_readme_filename: str
                ):
    config_streamlit = toml.load(Path(get_project_root()) / f'config/{config_streamlit_filename}')
    config_readme = toml.load(Path(get_project_root()) / f'config/{config_readme_filename}')
    return config_streamlit, config_readme

@st.cache()
def download_toy_dataset(url: str) -> pd.DataFrame:
    download = requests.get(url).content
    df = pd.read_csv(io.StringIO(download.decode('utf-8')))
    return df
