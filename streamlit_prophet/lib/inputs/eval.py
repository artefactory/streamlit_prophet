import streamlit as st


def input_metrics(readme: dict) -> dict:
    eval = dict()
    eval["metrics"] = st.multiselect(
        "Select evaluation metrics",
        ["MAPE", "SMAPE", "MSE", "RMSE", "MAE"],
        default=["MAPE", "RMSE"],
        help=readme["tooltips"]["metrics"],
    )
    return eval


def input_scope_eval(eval: dict, use_cv: bool, readme: dict) -> dict:
    if use_cv:
        eval["set"] = "Validation"
        eval["granularity"] = "cutoff"
    else:
        eval["set"] = st.selectbox(
            "Select evaluation set", ["Validation", "Training"], help=readme["tooltips"]["eval_set"]
        )
        eval["granularity"] = st.selectbox(
            "Select evaluation granularity",
            ["Global", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
            help=readme["tooltips"]["eval_granularity"],
        )
    eval["get_perf_on_agg_forecast"] = st.checkbox(
        "Get perf on aggregated forecast", value=False, help=readme["tooltips"]["choice_agg_perf"]
    )
    return eval
