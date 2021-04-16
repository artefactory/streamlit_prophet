from pathlib import Path
from lib.utils.path import list_files, get_project_root
from lib.utils.load import load_data
import streamlit as st


def input_dataset():
    # TODO: Ajouter la possibilité d'uploader son propre dataset
    data_filenames = [path.name for path in list_files(get_project_root() + '/data/', '*.*csv*')]
    filename = st.selectbox("Select a file", data_filenames)
    filepath = Path(get_project_root()) / 'data' / filename
    df = load_data(filepath)
    return df


def input_columns(config):
    date_col = st.text_input('Date column name', value=config["dataset"]["DATE_COLUMN"])
    target_col = st.text_input('Target column name', value=config["dataset"]["TARGET_COLUMN"])
    return date_col, target_col


def input_dimensions(df):
    eligible_cols = set(df.columns) - set(['ds', 'y'])
    dimensions = dict()
    if len(eligible_cols) > 0:
        dimensions_cols = st.multiselect("Choose dimensions", eligible_cols, default=[])
        for col in dimensions_cols:
            dimensions[col] = st.multiselect(f"Values to keep for {col}", df[col].unique(), default=df[col].unique())
    else:
        """Date and target are the only columns in your dataset, there are no dimensions."""
    return dimensions
