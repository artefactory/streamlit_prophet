import base64

import pandas as pd
import streamlit as st


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


def display_download_link(df: pd.DataFrame, filename: str, linkname: str, add_blank: bool = False):
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
