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