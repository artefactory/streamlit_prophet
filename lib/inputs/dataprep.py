import streamlit as st
from lib.utils.mapping import dayname_to_daynumber


def input_cleaning():
    del_days = st.multiselect("Remove days",
                              ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                               'Friday', 'Saturday', 'Sunday'], default=[])
    del_days = dayname_to_daynumber(del_days)
    del_zeros = st.checkbox('Delete rows where target = 0', True, key=1)
    del_negative = st.checkbox('Delete rows where target < 0', True, key=1)
    return del_days, del_zeros, del_negative

def input_dimensions(df, add_dimensions):
    dimensions = dict()
    if add_dimensions:
        eligible_cols = set(df.columns) - set(['ds', 'y'])
        if len(eligible_cols) > 0:
            dimensions_cols = st.multiselect("Choose dimensions", eligible_cols, default=[])
            for col in dimensions_cols:
                dimensions[col] = st.multiselect(f"Values to keep for {col}", df[col].unique(), default=df[col].unique())
        else:
            st.write("Date and target are the only columns in your dataset, there are no dimensions.")
    return dimensions