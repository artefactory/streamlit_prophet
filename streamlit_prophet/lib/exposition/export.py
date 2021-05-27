import base64

import pandas as pd
import streamlit as st
import toml


def get_df_download_link(df: pd.DataFrame, filename: str, linkname: str) -> str:
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


def get_config_download_link(config: dict, filename: str, linkname: str) -> str:
    """Creates a link to download the config as a toml file. Removes keys that should not be customized.

    Parameters
    ----------
    config : dict
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
    if "datasets" in config.keys():
        del config["datasets"]
    toml_string = toml.dumps(config)
    b64 = base64.b64encode(toml_string.encode()).decode()
    href = f'<a href="data:file/toml;base64,{b64}" download="{filename}">{linkname}</a>'
    return href


def display_download_link(
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
    st.markdown(get_df_download_link(df, filename, linkname), unsafe_allow_html=True)


def display_2_download_links(
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
        f"<p style='text-align: center;;'> {get_df_download_link(df1, filename1, linkname1)}</p>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<p style='text-align: center;;'> {get_df_download_link(df2, filename2, linkname2)}</p>",
        unsafe_allow_html=True,
    )


def display_config_download_links(
    config1: dict,
    filename1: str,
    linkname1: str,
    config2: dict,
    filename2: str,
    linkname2: str,
) -> None:
    """Displays a link to download a dataframe as a csv file.

    Parameters
    ----------
    config1 : dict
        First config file to export as toml.
    filename1 : str
        Name of the first exported file.
    linkname1 : str
        Text displayed in the streamlit app for the first link.
    config2 : dict
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
