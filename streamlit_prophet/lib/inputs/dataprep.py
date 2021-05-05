import pandas as pd
import streamlit as st
from streamlit_prophet.lib.utils.mapping import dayname_to_daynumber


def input_cleaning(resampling: dict, readme: dict) -> dict:
    # TODO : Ajouter une option "Remove holidays"
    cleaning = dict()
    if resampling["freq"][-1] in ["s", "H", "D"]:
        del_days = st.multiselect(
            "Remove days",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            default=[],
            help=readme["tooltips"]["remove_days"],
        )
        cleaning["del_days"] = dayname_to_daynumber(del_days)
    else:
        cleaning["del_days"] = []
    cleaning["del_zeros"] = st.checkbox(
        "Delete rows where target = 0", True, help=readme["tooltips"]["del_zeros"]
    )
    cleaning["del_negative"] = st.checkbox(
        "Delete rows where target < 0", True, help=readme["tooltips"]["del_negative"]
    )
    cleaning["log_transform"] = st.checkbox(
        "Target log transform", False, help=readme["tooltips"]["log_transform"]
    )
    return cleaning


def input_dimensions(df: pd.DataFrame, readme: dict) -> dict:
    dimensions = dict()
    eligible_cols = set(df.columns) - {"ds", "y"}
    if len(eligible_cols) > 0:
        dimensions_cols = st.multiselect(
            "Select dataset dimensions if any",
            list(eligible_cols),
            default=_autodetect_dimensions(df),
            help=readme["tooltips"]["dimensions"],
        )
        for col in dimensions_cols:
            values = list(df[col].unique())
            if st.checkbox(
                f"Keep all values for {col}", True, help=readme["tooltips"]["dimensions_filter"]
            ):
                dimensions[col] = values.copy()
            else:
                dimensions[col] = st.multiselect(
                    f"Values to keep for {col}",
                    values,
                    default=[values[0]],
                    help=readme["tooltips"]["dimensions_filter"],
                )
        dimensions["agg"] = st.selectbox(
            "Target aggregation function over dimensions",
            ["Mean", "Sum", "Max", "Min"],
            help=readme["tooltips"]["dimensions_agg"],
        )
    else:
        st.write("Date and target are the only columns in your dataset, there are no dimensions.")
        dimensions["agg"] = "Mean"
    return dimensions


def _autodetect_dimensions(df: pd.DataFrame) -> list:
    eligible_cols = set(df.columns) - {"ds", "y"}
    detected_cols = []
    for col in eligible_cols:
        values = df[col].value_counts()
        values = values.loc[values > 0].to_list()
        if (len(values) > 1) & (len(values) < 0.05 * len(df)):
            if max(values) / min(values) <= 20:
                detected_cols.append(col)
    return detected_cols


def input_resampling(df: pd.DataFrame, readme: dict) -> dict:
    resampling = dict()
    resampling["freq"] = _autodetect_freq(df)
    st.write(f"Frequency detected in dataset: {resampling['freq']}")
    resampling["resample"] = st.checkbox(
        "Resample my dataset", False, help=readme["tooltips"]["resample_choice"]
    )
    if resampling["resample"]:
        current_freq = resampling["freq"][-1]
        possible_freq_names = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
        possible_freq = [freq[0] for freq in possible_freq_names]
        current_freq_index = possible_freq.index(current_freq)
        if current_freq != "Y":
            new_freq = st.selectbox(
                "Select new frequency",
                possible_freq_names[current_freq_index + 1 :],
                help=readme["tooltips"]["resample_new_freq"],
            )
            resampling["freq"] = new_freq[0]
            resampling["agg"] = st.selectbox(
                "Target aggregation function when resampling",
                ["Mean", "Sum", "Max", "Min"],
                help=readme["tooltips"]["resample_agg"],
            )
        else:
            st.write("Frequency is already yearly, resampling is not possible.")
            resampling["resample"] = False
    return resampling


def _autodetect_freq(df: pd.DataFrame) -> str:
    min_delta = pd.Series(df["ds"]).diff().min()
    days = min_delta.days
    seconds = min_delta.seconds
    if days == 1:
        return "D"
    elif days < 1:
        if seconds >= 3600:
            return f"{round(seconds/3600)}H"
        else:
            return f"{seconds}s"
    elif days > 1:
        if days < 7:
            return f"{days}D"
        elif days < 28:
            return f"{round(days/7)}W"
        elif days < 90:
            return f"{round(days/30)}M"
        elif days < 365:
            return f"{round(days/90)}Q"
        else:
            return f"{round(days/365)}Y"
