import streamlit as st


def input_metrics() -> dict:
    eval = dict()
    eval['metrics'] = st.multiselect("Select evaluation metrics",
                                     ['MAPE', 'SMAPE', 'MSE', 'RMSE', 'MAE'],
                                     default=['MAPE', 'RMSE']
                                     )
    return eval


def input_scope_eval(eval: dict, use_cv: bool) -> dict:
    if use_cv:
        eval['set'] = 'Validation'
        eval['granularity'] = 'cutoff'
    else:
        eval['set'] = st.selectbox("Select evaluation set",
                                   ['Validation', 'Training'])
        eval['granularity'] = st.selectbox("Select evaluation granularity",
                                           ['Global', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'])
    eval['get_perf_on_agg_forecast'] = st.checkbox("Get perf on aggregated forecast", value=False)
    return eval
