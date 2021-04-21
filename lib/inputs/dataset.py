from lib.utils.load import download_dataset
import pandas as pd
import streamlit as st


def input_dataset(config):
    if st.checkbox('Upload my own dataset', False):
        upload = st.file_uploader("Upload a csv file", type='csv')
        if upload:
            df = pd.read_csv(upload)
        else:
            st.stop()
    else:
        dataset_name = st.selectbox("Or select a toy dataset", list(config['datasets'].keys()))
        df = download_dataset(config['datasets'][dataset_name])
    return df


def input_columns(df: pd.DataFrame):
    date_col = st.selectbox("Date column", list(df.columns))
    target_col = st.selectbox("Target column", list(set(df.columns) - set([date_col])))
    return date_col, target_col
