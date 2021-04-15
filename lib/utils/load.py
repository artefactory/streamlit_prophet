import pandas as pd
import streamlit as st
from loguru import logger

#@st.cache
def load_data(filepath: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    - filepath: str
    Returns
    -------
    pd.DataFrame
        A dataFrame containing the components in columns, for each date (rows)
    """
    try:
        return pd.read_csv(filepath)
    except ValueError as e:
        print(f"{e}, File not found.")
        return None