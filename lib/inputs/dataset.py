from pathlib import Path
from lib.utils.path import list_files, get_project_root
from lib.utils.load import load_data
import pandas as pd
import streamlit as st


def input_dataset():
    # TODO: Ajouter la possibilité d'uploader son propre dataset
    data_filenames = [path.name for path in list_files(get_project_root() + '/data/', '*.*csv*')]
    filename = st.selectbox("Select a file", data_filenames)
    filepath = Path(get_project_root()) / 'data' / filename
    df = load_data(filepath)
    return df


def input_columns(df: pd.DataFrame):
    date_col = st.selectbox("Date column", list(df.columns))
    target_col = st.selectbox("Target column", list(set(df.columns) - set([date_col])))
    return date_col, target_col
