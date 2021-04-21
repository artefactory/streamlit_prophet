from pathlib import Path
from lib.utils.path import list_files, get_project_root
from lib.utils.load import load_dataset
import pandas as pd
import streamlit as st


def input_dataset():
    if st.checkbox('Upload my own dataset', False):
        upload = st.file_uploader("Upload a csv file", type='csv')
        if upload:
            df = load_dataset(upload)
        else:
            st.stop()
    else:
        data_filenames = [path.name for path in list_files(get_project_root() + '/data/', '*.*csv*')]
        filename = st.selectbox("Select a file", data_filenames)
        filepath = Path(get_project_root()) / 'data' / filename
        df = load_dataset(filepath)
    return df


def input_columns(df: pd.DataFrame):
    date_col = st.selectbox("Date column", list(df.columns))
    target_col = st.selectbox("Target column", list(set(df.columns) - set([date_col])))
    return date_col, target_col
