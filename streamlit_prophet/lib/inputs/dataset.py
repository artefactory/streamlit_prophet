from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st
from streamlit_prophet.lib.exposition.export import display_config_download_links
from streamlit_prophet.lib.utils.load import download_toy_dataset, load_custom_config, load_dataset


def input_dataset(
    config: Dict[Any, Any], readme: Dict[Any, Any], instructions: Dict[Any, Any]
) -> Tuple[pd.DataFrame, Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
    """Lets the user decide whether to upload a dataset or download a toy dataset.

    Parameters
    ----------
    config : Dict
        Lib config dictionary containing information about toy datasets (download links).
    readme : Dict
        Dictionary containing tooltips to guide user's choices.
    instructions : Dict
        Dictionary containing instructions to provide a custom config.

    Returns
    -------
    pd.DataFrame
        Selected dataset loaded into a dataframe.
    dict
        Loading options selected by user (upload or download, dataset name if download).
    dict
        Lib configuration dictionary.
    dict
        Dictionary containing all datasets.
    """
    load_options, datasets = dict(), dict()
    load_options["toy_dataset"] = st.checkbox(
        "Load a toy dataset", True, help=readme["tooltips"]["upload_choice"]
    )
    if load_options["toy_dataset"]:
        dataset_name = st.selectbox(
            "Select a toy dataset",
            options=list(config["datasets"].keys()),
            format_func=lambda x: config["datasets"][x]["name"],
            help=readme["tooltips"]["toy_dataset"],
        )
        df = download_toy_dataset(config["datasets"][dataset_name]["url"])
        load_options["dataset"] = dataset_name
        load_options["date_format"] = config["dataprep"]["date_format"]
        load_options["separator"] = ","
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
            with st.sidebar.expander("Configuration", expanded=True):
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
    datasets["uploaded"] = df.copy()
    return df, load_options, config, datasets


def input_columns(
    config: Dict[Any, Any], readme: Dict[Any, Any], df: pd.DataFrame, load_options: Dict[Any, Any]
) -> Tuple[str, str]:
    """Lets the user specify date and target column names.

    Parameters
    ----------
    config : Dict
        Lib config dictionary containing information about toy datasets (date and target column names).
    readme : Dict
        Dictionary containing tooltips to guide user's choices.
    df : pd.DataFrame
        Loaded dataset.
    load_options : Dict
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
            sorted(df.columns)
            if config["columns"]["date"] in ["false", False]
            else [config["columns"]["date"]],
            help=readme["tooltips"]["date_column"],
        )
        target_col = st.selectbox(
            "Target column",
            sorted(set(df.columns) - {date_col})
            if config["columns"]["target"] in ["false", False]
            else [config["columns"]["target"]],
            help=readme["tooltips"]["target_column"],
        )
    return date_col, target_col


def input_future_regressors(
    datasets: Dict[Any, Any],
    dates: Dict[Any, Any],
    params: Dict[Any, Any],
    dimensions: Dict[Any, Any],
    load_options: Dict[Any, Any],
    date_col: str,
) -> pd.DataFrame:
    """Adds future regressors dataframe in datasets dictionary's values.

    Parameters
    ----------
    datasets : Dict
        Dictionary storing all dataframes.
    dates : Dict
        Dictionary containing future forecasting dates information.
    params : Dict
        Dictionary containing all model parameters and list of selected regressors.
    dimensions : Dict
        Dictionary containing dimensions information.
    load_options : Dict
        Loading options selected by user (including csv delimiter).
    date_col : str
        Name of date column.

    Returns
    -------
    dict
        The datasets dictionary containing future regressors dataframe.
    """
    if len(params["regressors"].keys()) > 0:
        regressors_col = list(params["regressors"].keys())
        start, end = dates["forecast_start_date"], dates["forecast_end_date"]
        tooltip = (
            f"Please upload a csv file with delimiter '{load_options['separator']}' "
            "and the same format as input dataset, ie with the following specifications: \n"
        )
        tooltip += (
            f"- Date column named `{date_col}`, going from **{start.strftime('%Y-%m-%d')}** "
            f"to **{end.strftime('%Y-%m-%d')}** at the same frequency as input dataset "
            f"and at format **{load_options['date_format']}**. \n"
        )
        dimensions_col = [col for col in dimensions.keys() if col != "agg"]
        if len(dimensions_col) > 0:
            if len(dimensions_col) > 1:
                tooltip += (
                    f"- Columns with the following names for dimensions: `{', '.join(dimensions_col[:-1])}, "
                    f"{dimensions_col[-1]}`. \n"
                )
            else:
                tooltip += f"- Dimension column named `{dimensions_col[0]}`. \n"
        if len(regressors_col) > 1:
            tooltip += (
                f"- Columns with the following names for regressors: `{', '.join(regressors_col[:-1])}, "
                f"{regressors_col[-1]}`."
            )
        else:
            tooltip += f"- Regressor column named `{regressors_col[0]}`."
        regressors_file = st.file_uploader(
            "Upload a csv file for regressors", type="csv", help=tooltip
        )
        if regressors_file:
            datasets["future_regressors"] = load_dataset(regressors_file, load_options)
    else:
        st.write("There are no regressors selected.")
    return datasets
