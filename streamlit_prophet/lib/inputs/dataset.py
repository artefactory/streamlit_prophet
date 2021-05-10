from typing import Tuple

import pandas as pd
import streamlit as st
from streamlit_prophet.lib.utils.load import download_toy_dataset, load_dataset


def input_dataset(config: dict, readme: dict) -> Tuple[pd.DataFrame, dict]:
    """Lets the user decide whether to upload a dataset or download a toy dataset.

    Parameters
    ----------
    config : dict
        Lib config dictionary containing information about toy datasets (download links).
    readme : dict
        Dictionary containing tooltips to guide user's choices.

    Returns
    -------
    pd.DataFrame
        Selected dataset loaded into a dataframe.
    dict
        Loading options selected by user (upload or download, dataset name if download).
    """
    load_options = dict()
    load_options["toy_dataset"] = st.checkbox(
        "Load a toy dataset", True, help=readme["tooltips"]["upload_choice"]
    )
    if load_options["toy_dataset"]:
        dataset_name = st.selectbox(
            "Select a toy dataset",
            list(config["datasets"].keys()),
            help=readme["tooltips"]["toy_dataset"],
        )
        df = download_toy_dataset(config["datasets"][dataset_name]["url"])
        load_options["dataset"] = dataset_name
    else:
        file = st.file_uploader(
            "Upload a csv file", type="csv", help=readme["tooltips"]["dataset_upload"]
        )
        if file:
            df = load_dataset(file)
        else:
            st.stop()
    return df, load_options


def input_columns(
    config: dict, readme: dict, df: pd.DataFrame, load_options: dict
) -> Tuple[str, str]:
    """Lets the user specify date and target column names.

    Parameters
    ----------
    config : dict
        Lib config dictionary containing information about toy datasets (date and target column names).
    readme : dict
        Dictionary containing tooltips to guide user's choices.
    df : pd.DataFrame
        Loaded dataset.
    load_options : dict
        Loading options selected by user (upload or download, dataset name if download).

    Returns
    -------
    str
        Date column name.
    str
        Target column name.
    """
    if load_options["toy_dataset"]:
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
    else:
        date_col = st.selectbox(
            "Date column", list(df.columns), help=readme["tooltips"]["date_column"]
        )
        target_col = st.selectbox(
            "Target column",
            list(set(df.columns) - {date_col}),
            help=readme["tooltips"]["target_column"],
        )
    return date_col, target_col
