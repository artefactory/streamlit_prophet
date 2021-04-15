from pathlib import Path
from lib.utils.path import list_files, get_project_root
from lib.utils.load import load_data
import streamlit as st


def input_dataset():
    #TODO: Ajouter la possibilité d'uploader son propre dataset
    data_filenames = [path.name for path in list_files(get_project_root() + '/data/', '*.*csv*')]
    filename = st.selectbox("Select a file:", data_filenames)
    filepath = Path(get_project_root()) / 'data' / filename
    df = load_data(filepath)
    return df

def input_columns(config):
    date_col = st.text_input('Date column name', value=config["dataset"]["DATE_COLUMN"])
    target_col = st.text_input('Target column name', value=config["dataset"]["TARGET_COLUMN"])
    return date_col, target_col