from typing import Any, Dict, List

import base64
import io
from base64 import b64encode
from zipfile import ZipFile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import toml


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
    col1, col2 = st.beta_columns(2)
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
    col1, col2 = st.beta_columns(2)
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
    zip_path = "report/streamlit_prophet.zip"
    zipObj = ZipFile(zip_path, "w")
    # Save plots and data
    for x in report:
        if x["type"] == "plot":
            file_path = f"report/plots/{x['name']}.html"
            x["object"].write_html(file_path)
        if x["type"] == "dataset":
            file_path = f"report/data/{x['name']}.csv"
            x["object"].to_csv(file_path, index=False)
        zipObj.write(file_path)
    # Save default config
    default_config = config.copy()
    if "datasets" in default_config.keys():
        del default_config["datasets"]
    file_path = f"report/config/default_config.toml"
    with open(file_path, "w") as toml_file:
        toml.dump(default_config, toml_file)
    zipObj.write(file_path)
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
    file_path = f"report/config/user_specifications.toml"
    with open(file_path, "w") as toml_file:
        toml.dump(all_specs, toml_file)
    zipObj.write(file_path)
    # Close zip file
    zipObj.close()
    return zip_path


def display_save_experiment_link(zip_path: str) -> None:
    """Displays a link to export the report as a zip file.

    Parameters
    ----------
    zip_path: str
        Path of the zip file.
    """
    with open(zip_path, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f"<a href=\"data:file/zip;base64,{b64}\" download='{zip_path}'>Download report</a>"
    st.markdown(href, unsafe_allow_html=True)


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
    readme: Dict[Any, Any],
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
    readme : Dict
        Dictionary containing explanations on the button.
    """
    col1, col2 = st.beta_columns([1, 4])
    if col1.button("Save experiment", help=readme["tooltips"]["save_experiment_button"]):
        with col2:
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
                display_save_experiment_link(zip_path)
