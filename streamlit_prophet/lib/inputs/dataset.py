from typing import Tuple

import pandas as pd
import streamlit as st
from streamlit_prophet.lib.exposition.export import display_config_download_links
from streamlit_prophet.lib.utils.load import download_toy_dataset, load_custom_config, load_dataset


def input_dataset(config: dict, readme: dict, instructions: dict) -> Tuple[pd.DataFrame, dict]:
    """Lets the user decide whether to upload a dataset or download a toy dataset.

    Parameters
    ----------
    config : dict
        Lib config dictionary containing information about toy datasets (download links).
    readme : dict
        Dictionary containing tooltips to guide user's choices.
    instructions : dict
        Dictionary containing instructions to provide a custom config.

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
        load_options["date_format"] = config["dataprep"]["date_format"]
    else:
        file = st.file_uploader(
            "Upload a csv file", type="csv", help=readme["tooltips"]["dataset_upload"]
        )
        load_options["separator"] = st.selectbox(
            "What is the separator?", [",", ";", "|"], help=readme["tooltips"]["separator"]
        )
        load_options["date_format"] = st.text_input(
            "What is the date format?",
            config["dataprep"]["date_format"],
            help=readme["tooltips"]["date_format"],
        )
        if st.checkbox(
            "Upload my own config file", False, help=readme["tooltips"]["custom_config_choice"]
        ):
            with st.sidebar.beta_expander("Configuration", expanded=True):
                display_config_download_links(
                    config,
                    "config.toml",
                    "Template",
                    instructions,
                    "instructions.toml",
                    "Instructions",
                )
                config_file = st.file_uploader(
                    "Upload custom config", type="toml", help=readme["tooltips"]["custom_config"]
                )
                if config_file:
                    config = load_custom_config(config_file)
                else:
                    st.stop()
        if file:
            df = load_dataset(file, load_options)
        else:
            st.stop()
    return df, load_options, config


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
            "Date column",
            list(df.columns)
            if config["columns"]["date"] in ["false", False]
            else [config["columns"]["date"]],
            help=readme["tooltips"]["date_column"],
        )
        target_col = st.selectbox(
            "Target column",
            list(set(df.columns) - {date_col})
            if config["columns"]["target"] in ["false", False]
            else [config["columns"]["target"]],
            help=readme["tooltips"]["target_column"],
        )
    return date_col, target_col
