from typing import Any, Dict

import streamlit as st


def input_metrics(readme: Dict[Any, Any], config: Dict[Any, Any]) -> Dict[Any, Any]:
    """Lets the user select evaluation metrics to be used for model evaluation.

    Parameters
    ----------
    readme : Dict
        Dictionary containing tooltips to guide user's choices.
    config : Dict
        Lib config containing the default list of metrics to use for evaluation.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics information.
    """
    eval = dict()
    eval["metrics"] = st.multiselect(
        "Select evaluation metrics",
        ["MAPE", "SMAPE", "MSE", "RMSE", "MAE"],
        default=config["metrics"]["default"]["selection"],
        help=readme["tooltips"]["metrics"],
    )
    return eval


def input_scope_eval(eval: Dict[Any, Any], use_cv: bool, readme: Dict[Any, Any]) -> Dict[Any, Any]:
    """Lets the user define the scope of model evaluation (granularity, evaluation set).

    Parameters
    ----------
    eval : Dict
        Dictionary containing evaluation metrics information.
    use_cv : bool
        Whether or not cross-validation is used.
    readme : Dict
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
