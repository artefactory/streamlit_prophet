import streamlit as st


def input_metrics(eval: dict) -> dict:
    eval['metrics'] = st.multiselect("Select evaluation metrics",
                                     ['MAPE', 'SMAPE', 'MSE', 'RMSE', 'MAE'],
                                     default=['MAPE', 'RMSE']
                                     )
    return eval


def input_scope_eval(eval: dict) -> dict:
    eval['set'] = st.selectbox("Select evaluation set",
                               ['Validation', 'Training'])
    eval['granularity'] = st.selectbox("Select evaluation granularity",
                                       ['Global', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'])
    eval['get_perf_on_agg_forecast'] = st.checkbox("Get perf on aggregated forecast", value=False)
    return eval
