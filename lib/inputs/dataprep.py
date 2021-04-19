import streamlit as st
from lib.utils.mapping import dayname_to_daynumber


def input_cleaning(cleaning_options: dict):
    del_days = st.multiselect("Remove days",
                              ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                               'Friday', 'Saturday', 'Sunday'], default=[])
    cleaning_options['del_days'] = dayname_to_daynumber(del_days)
    cleaning_options['del_zeros'] = st.checkbox('Delete rows where target = 0', True, key=1)
    cleaning_options['del_negative'] = st.checkbox('Delete rows where target < 0', True, key=1)
    cleaning_options['log_transform'] = st.checkbox('Target log transform', False, key=1)
    return cleaning_options


def input_dimensions(df):
    dimensions = dict()
    eligible_cols = set(df.columns) - set(['ds', 'y'])
    if len(eligible_cols) > 0:
        dimensions_cols = st.multiselect("Select dataset dimensions if any",
                                         list(eligible_cols),
                                         default=autodetect_dimensions(df)
                                         )
        for col in dimensions_cols:
            # TODO: Ajouter checkbox "keep all values"
            values = list(df[col].unique())
            dimensions[col] = st.multiselect(f"Values to keep for {col}", values, default=[values[0]])
    else:
        st.write("Date and target are the only columns in your dataset, there are no dimensions.")
    return dimensions


def autodetect_dimensions(df):
    eligible_cols = set(df.columns) - set(['ds', 'y'])
    detected_cols = []
    for col in eligible_cols:
        values = df[col].value_counts()
        values = values.loc[values > 0].to_list()
        if max(values) / min(values) <= 20:
            detected_cols.append(col)
    return detected_cols