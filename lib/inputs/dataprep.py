import streamlit as st
import pandas as pd
from lib.utils.mapping import dayname_to_daynumber


def input_cleaning(resampling: dict):
    # TODO : Ajouter une option "Remove holidays"
    cleaning = dict()
    if resampling['freq'][-1] in ['s', 'H', 'D']:
        del_days = st.multiselect("Remove days",
                                  ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                   'Friday', 'Saturday', 'Sunday'], default=[])
        cleaning['del_days'] = dayname_to_daynumber(del_days)
    else:
        cleaning['del_days'] = []
    cleaning['del_zeros'] = st.checkbox('Delete rows where target = 0', True)
    cleaning['del_negative'] = st.checkbox('Delete rows where target < 0', True)
    if cleaning['del_zeros'] & cleaning['del_negative']:
        cleaning['log_transform'] = st.checkbox('Target log transform', False)
    else:
        cleaning['log_transform'] = False
    return cleaning


def input_dimensions(df):
    dimensions = dict()
    eligible_cols = set(df.columns) - set(['ds', 'y'])
    if len(eligible_cols) > 0:
        dimensions_cols = st.multiselect("Select dataset dimensions if any",
                                         list(eligible_cols),
                                         default=_autodetect_dimensions(df)
                                         )
        for col in dimensions_cols:
            values = list(df[col].unique())
            if st.checkbox(f'Keep all values for {col}', True, key=1):
                dimensions[col] = values.copy()
            else:
                dimensions[col] = st.multiselect(f"Values to keep for {col}", values, default=[values[0]])
    else:
        st.write("Date and target are the only columns in your dataset, there are no dimensions.")
    return dimensions


def _autodetect_dimensions(df):
    eligible_cols = set(df.columns) - set(['ds', 'y'])
    detected_cols = []
    for col in eligible_cols:
        values = df[col].value_counts()
        values = values.loc[values > 0].to_list()
        if max(values) / min(values) <= 20:
            detected_cols.append(col)
    return detected_cols


def input_resampling(df):
    resampling = dict()
    resampling['freq'] = _autodetect_freq(df)
    st.write(f"Frequency of dataset: {resampling['freq']}")
    resampling['resample'] = st.checkbox('Resample my dataset', False)
    if resampling['resample']:
        current_freq = resampling['freq'][-1]
        possible_freq_names = ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
        possible_freq = [freq[0] for freq in possible_freq_names]
        current_freq_index = possible_freq.index(current_freq)
        if current_freq != 'Y':
            new_freq = st.selectbox("Select new frequency", possible_freq_names[current_freq_index+1:])
            resampling['freq'] = new_freq[0]
        else:
            st.write('Frequency is already yearly, resampling is not possible.')
    return resampling


def _autodetect_freq(df):
    min_delta = pd.Series(df['ds']).diff().min()
    days = min_delta.days
    seconds = min_delta.seconds
    if days == 1:
        return 'D'
    elif days < 1:
        if seconds >= '3600':
            return f'{round(seconds/60)}H'
        else:
            return f'{seconds}s'
    elif days > 1:
        if days < 7:
            return f'{days}D'
        elif days < 28:
            return f'{round(days/7)}W'
        elif days < 90:
            return f'{round(days/30)}M'
        elif days < 365:
            return f'{round(days/90)}Q'
        else:
            return f'{round(days/365)}Y'



