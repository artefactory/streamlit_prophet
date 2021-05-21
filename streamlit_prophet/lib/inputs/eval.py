import streamlit as st


def input_metrics(readme: dict) -> dict:
    """Lets the user select evaluation metrics to be used for model evaluation.

    Parameters
    ----------
    readme : dict
        Dictionary containing tooltips to guide user's choices.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics information.
    """
    eval = dict()
    eval["metrics"] = st.multiselect(
        "Select evaluation metrics",
        ["MAPE", "SMAPE", "MSE", "RMSE", "MAE"],
        default=["MAPE", "RMSE"],
        help=readme["tooltips"]["metrics"],
    )
    return eval


def input_scope_eval(eval: dict, use_cv: bool, readme: dict) -> dict:
    """Lets the user define the scope of model evaluation (granularity, evaluation set).

    Parameters
    ----------
    eval : dict
        Dictionary containing evaluation metrics information.
    use_cv : bool
        Whether or not cross-validation is used.
    readme : dict
        Dictionary containing tooltips to guide user's choices.

    Returns
    -------
    dict
        Dictionary containing information about the scope of model evaluation (granularity, evaluation set).
    """
    if use_cv:
        eval["set"] = "Validation"
        eval["granularity"] = "cutoff"
    else:
        eval["set"] = st.selectbox(
            "Select evaluation set", ["Validation", "Training"], help=readme["tooltips"]["eval_set"]
        )
        eval["granularity"] = st.selectbox(
            "Select evaluation granularity",
            ["Daily", "Day of Week", "Weekly", "Monthly", "Quarterly", "Yearly", "Global"],
            help=readme["tooltips"]["eval_granularity"],
        )
    eval["get_perf_on_agg_forecast"] = st.checkbox(
        "Get perf on aggregated forecast", value=False, help=readme["tooltips"]["choice_agg_perf"]
    )
    return eval
