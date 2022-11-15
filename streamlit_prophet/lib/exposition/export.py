from typing import Any, Dict, List

import base64
import io
import re
import uuid
from base64 import b64encode
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import toml
from streamlit_prophet.lib.utils.load import get_project_root


def get_dataframe_download_link(df: pd.DataFrame, filename: str, linkname: str) -> str:
    """Creates a link to download a dataframe as a csv file.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to export.
    filename : str
        Name of the exported file.
    linkname : str
        Text displayed in the streamlit app.

    Returns
    -------
    str
        Download link.
    """
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{linkname}</a>'
    return href


def get_config_download_link(config: Dict[Any, Any], filename: str, linkname: str) -> str:
    """Creates a link to download the config as a toml file. Removes keys that should not be customized.

    Parameters
    ----------
    config : Dict
        Config file to export as toml.
    filename : str
        Name of the exported file.
    linkname : str
        Text displayed in the streamlit app.

    Returns
    -------
    str
        Download link.
    """
    config_template = config.copy()
    if "datasets" in config_template.keys():
        del config_template["datasets"]
    toml_string = toml.dumps(config)
    b64 = base64.b64encode(toml_string.encode()).decode()
    href = f'<a href="data:file/toml;base64,{b64}" download="{filename}">{linkname}</a>'
    return href


def get_plotly_download_link(fig: go.Figure, filename: str, linkname: str) -> str:
    """Creates a link to download a dataframe as a html file.

    Parameters
    ----------
    fig : go.Figure
        Plotly go figure to export.
    filename : str
        Name of the exported file.
    linkname : str
        Text displayed in the streamlit app.

    Returns
    -------
    str
        Download link.
    """
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_bytes = buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()
    href = f'<a href="data:text/html;base64,{encoded}" download="{filename}.html">{linkname}</a>'
    return href


def display_dataframe_download_link(
    df: pd.DataFrame, filename: str, linkname: str, add_blank: bool = False
) -> None:
    """Displays a link to download a dataframe as a csv file.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to export.
    filename : str
        Name of the exported file.
    linkname : str
        Text displayed in the streamlit app.
    add_blank : str
        Whether or not to add a blank before the link in streamlit app.
    """
    if add_blank:
        st.write("")
    st.markdown(get_dataframe_download_link(df, filename, linkname), unsafe_allow_html=True)


def display_2_dataframe_download_links(
    df1: pd.DataFrame,
    filename1: str,
    linkname1: str,
    df2: pd.DataFrame,
    filename2: str,
    linkname2: str,
    add_blank: bool = False,
) -> None:
    """Displays a link to download a dataframe as a csv file.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe to export.
    filename1 : str
        Name of the first exported file.
    linkname1 : str
        Text displayed in the streamlit app for the first link.
    df2 : pd.DataFrame
        Second dataframe to export.
    filename2 : str
        Name of the second exported file.
    linkname2 : str
        Text displayed in the streamlit app for the second link.
    add_blank : str
        Whether or not to add a blank before the link in streamlit app.
    """
    if add_blank:
        st.write("")
    col1, col2 = st.columns(2)
    col1.markdown(
        f"<p style='text-align: center;;'> {get_dataframe_download_link(df1, filename1, linkname1)}</p>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<p style='text-align: center;;'> {get_dataframe_download_link(df2, filename2, linkname2)}</p>",
        unsafe_allow_html=True,
    )


def display_config_download_links(
    config1: Dict[Any, Any],
    filename1: str,
    linkname1: str,
    config2: Dict[Any, Any],
    filename2: str,
    linkname2: str,
) -> None:
    """Displays a link to download a dataframe as a csv file.

    Parameters
    ----------
    config1 : Dict
        First config file to export as toml.
    filename1 : str
        Name of the first exported file.
    linkname1 : str
        Text displayed in the streamlit app for the first link.
    config2 : Dict
        Second config file to export as toml.
    filename2 : str
        Name of the second exported file.
    linkname2 : str
        Text displayed in the streamlit app for the second link.
    """
    col1, col2 = st.columns(2)
    col1.markdown(
        f"<p style='text-align: center;;'> {get_config_download_link(config1, filename1, linkname1)}</p>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<p style='text-align: center;;'> {get_config_download_link(config2, filename2, linkname2)}</p>",
        unsafe_allow_html=True,
    )


def display_plotly_download_link(
    fig: go.Figure, filename: str, linkname: str, add_blank: bool = False
) -> None:
    """Displays a link to export a plotly graph in html.

    Parameters
    ----------
    fig : go.Figure
        Plotly go figure to export.
    filename : str
        Name of the exported file.
    linkname : str
        Text displayed in the streamlit app.
    add_blank : str
        Whether or not to add a blank before the link in streamlit app.
    """
    if add_blank:
        st.write("")
    st.markdown(get_plotly_download_link(fig, filename, linkname), unsafe_allow_html=True)


def create_report_zip_file(
    report: List[Dict[str, Any]],
    config: Dict[Any, Any],
    use_cv: bool,
    make_future_forecast: bool,
    evaluate: bool,
    cleaning: Dict[Any, Any],
    resampling: Dict[Any, Any],
    params: Dict[Any, Any],
    dates: Dict[Any, Any],
    date_col: str,
    target_col: str,
    dimensions: Dict[Any, Any],
) -> str:
    """Saves locally all report components in a zip file.

    Parameters
    ----------
    report: List[Dict[str, Any]]
        List of all report components.
    config : Dict
        Lib configuration dictionary.
    use_cv : bool
        Whether or not cross-validation is used.
    make_future_forecast : bool
        Whether or not to make a forecast on future dates.
    evaluate : bool
        Whether or not to do a model evaluation.
    cleaning : Dict
        Dataset cleaning specifications.
    resampling : Dict
        Dataset resampling specifications.
    params : Dict
        Model parameters.
    dates : Dict
        Dictionary containing all relevant dates for training and forecasting.
    date_col : str
        Name of date column.
    target_col : str
        Name of target column.
    dimensions : Dict
        Dictionary containing dimensions information.

    Returns
    -------
    str
        Path of the zip file.
    """
    # Create zip file
    zip_path = "experiment.zip"
    zipObj = ZipFile(zip_path, "w")
    report_name = f"report_{datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss')}"
    # Save plots and data
    for x in report:
        if x["type"] == "plot":
            file_name = f"{report_name}/plots/{x['name']}.html"
            file_path = _get_file_path(file_name)
            x["object"].write_html(file_path)
        if x["type"] == "dataset":
            file_name = f"{report_name}/data/{x['name']}.csv"
            file_path = _get_file_path(file_name)
            x["object"].to_csv(file_path, index=False)
        zipObj.write(file_path, arcname=file_name)
    # Save default config
    default_config = config.copy()
    if "datasets" in default_config.keys():
        del default_config["datasets"]
    file_name = f"{report_name}/config/default_config.toml"
    file_path = _get_file_path(file_name)
    with open(file_path, "w") as toml_file:
        toml.dump(default_config, toml_file)
    zipObj.write(file_path, arcname=file_name)
    # Save user specifications
    all_specs = {
        "model_params": params,
        "dates": dates,
        "columns": {"date": date_col, "target": target_col},
        "filtering": dimensions,
        "cleaning": cleaning,
        "resampling": resampling,
        "actions": {
            "evaluate": evaluate,
            "use_cv": use_cv,
            "make_future_forecast": make_future_forecast,
        },
    }
    file_name = f"{report_name}/config/user_specifications.toml"
    file_path = _get_file_path(file_name)
    with open(file_path, "w") as toml_file:
        toml.dump(all_specs, toml_file)
    zipObj.write(file_path, arcname=file_name)
    # Close zip file
    zipObj.close()
    return zip_path


def _get_file_path(file_name: str) -> str:
    """Returns the full path of a file to include in the zip file.

    Parameters
    ----------
    file_name: str
        Short file name.

    Returns
    -------
    str
        Full path.
    """
    return str(Path(get_project_root()) / f"report/{'/'.join(file_name.split('/')[1:])}")


def create_save_experiment_button(zip_path: str) -> None:
    """Displays a link to export the report as a zip file.

    Parameters
    ----------
    zip_path: str
        Path of the zip file.
    """

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub(r"\d+", "", button_uuid)

    with open(zip_path, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = (
            f"<a download='{zip_path}' id='{button_id}' href=\"data:file/zip;base64,{b64}\" >Save "
            f"experiment</a><br></br> "
        )

    color1 = "rgb(255, 0, 102)"
    color2 = "rgb(0, 34, 68)"
    color3 = "rgb(255, 255, 255)"
    custom_css = f"""
            <style>
                #{button_id} {{
                    background-color: {color1};
                    color: {color3};
                    padding: 0.45em 0.58em;
                    position: relative;
                    text-decoration: none;
                    border-radius: 5px;
                    border-width: 2px;
                    border-style: solid;
                    border-color: {color3};
                    border-image: initial;
                }}
                #{button_id}:hover {{
                    border-color: {color2};
                    color: {color2};
                }}
                #{button_id}:active {{
                    box-shadow: none;
                    background-color: {color3};
                    border-color: {color1};
                    color: {color1};
                    }}
            </style> """

    st.markdown(
        f"<p style='text-align: center;;'> {custom_css + href} </p>", unsafe_allow_html=True
    )


def display_save_experiment_button(
    report: List[Dict[str, Any]],
    config: Dict[Any, Any],
    use_cv: bool,
    make_future_forecast: bool,
    evaluate: bool,
    cleaning: Dict[Any, Any],
    resampling: Dict[Any, Any],
    params: Dict[Any, Any],
    dates: Dict[Any, Any],
    date_col: str,
    target_col: str,
    dimensions: Dict[Any, Any],
) -> None:
    """Saves locally all report components in a zip file.

    Parameters
    ----------
    report: List[Dict[str, Any]]
        List of all report components.
    config : Dict
        Lib configuration dictionary.
    use_cv : bool
        Whether or not cross-validation is used.
    make_future_forecast : bool
        Whether or not to make a forecast on future dates.
    evaluate : bool
        Whether or not to do a model evaluation.
    cleaning : Dict
        Dataset cleaning specifications.
    resampling : Dict
        Dataset resampling specifications.
    params : Dict
        Model parameters.
    dates : Dict
        Dictionary containing all relevant dates for training and forecasting.
    date_col : str
        Name of date column.
    target_col : str
        Name of target column.
    dimensions : Dict
        Dictionary containing dimensions information.
    """
    with st.spinner("Saving config, plots and data..."):
        zip_path = create_report_zip_file(
            report,
            config,
            use_cv,
            make_future_forecast,
            evaluate,
            cleaning,
            resampling,
            params,
            dates,
            date_col,
            target_col,
            dimensions,
        )
        create_save_experiment_button(zip_path)


def display_links(repo_link: str, article_link: str) -> None:
    """Displays a repository and app links.

    Parameters
    ----------
    repo_link : str
        Link of git repository.
    article_link : str
        Link of medium article.
    """
    col1, col2 = st.sidebar.columns(2)
    col1.markdown(
        f"<a style='display: block; text-align: center;' href={repo_link}>Source code</a>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<a style='display: block; text-align: center;' href={article_link}>App introduction</a>",
        unsafe_allow_html=True,
    )
