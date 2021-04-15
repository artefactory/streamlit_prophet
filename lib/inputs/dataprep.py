import streamlit as st

def input_cleaning():
    del_days = st.multiselect("Remove days:",
                              ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                               'Friday', 'Saturday', 'Sunday'], default=[])
    del_days = dayname_to_daynumber(del_days)
    del_zeros = st.checkbox('Delete rows where target = 0', True, key=1)
    del_negative = st.checkbox('Delete rows where target < 0', True, key=1)
    return del_days, del_zeros, del_negative

def dayname_to_daynumber(days: list):
    mapping = {'Monday': 0,
               'Tuesday': 1,
               'Wednesday': 2,
               'Thursday': 3,
               'Friday': 4,
               'Saturday': 5,
               'Sunday': 6
               }
    return [mapping[day] for day in days]