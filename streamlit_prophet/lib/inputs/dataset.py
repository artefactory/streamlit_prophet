import pandas as pd
import streamlit as st
from streamlit_prophet.lib.utils.load import download_toy_dataset, load_dataset


def input_dataset(config: dict, readme: dict):
    load_options = dict()
    load_options["upload"] = st.checkbox(
        "Upload my own dataset", False, help=readme["tooltips"]["upload_choice"]
    )
    if load_options["upload"]:
        file = st.file_uploader(
            "Upload a csv file", type="csv", help=readme["tooltips"]["dataset_upload"]
        )
        if file:
            df = load_dataset(file)
        else:
            st.stop()
    else:
        dataset_name = st.selectbox(
            "Or select a toy dataset",
            list(config["datasets"].keys()),
            help=readme["tooltips"]["toy_dataset"],
        )
        df = download_toy_dataset(config["datasets"][dataset_name]["url"])
        load_options["dataset"] = dataset_name
    return df, load_options


def input_columns(config: dict, readme: dict, df: pd.DataFrame, load_options: dict):
    if load_options["upload"]:
        date_col = st.selectbox(
            "Date column", list(df.columns), help=readme["tooltips"]["date_column"]
        )
        target_col = st.selectbox(
            "Target column",
            list(set(df.columns) - {date_col}),
            help=readme["tooltips"]["target_column"],
        )
    else:
        date_col = st.selectbox(
            "Date column",
            [config["datasets"][load_options["dataset"]]["date"]],
            help=readme["tooltips"]["date_column"],
        )
        target_col = st.selectbox(
            "Target column",
            [config["datasets"][load_options["dataset"]]["target"]],
            help=readme["tooltips"]["target_column"],
        )
    return date_col, target_col
