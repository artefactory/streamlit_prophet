from typing import Any, Dict, Tuple

from datetime import timedelta

import numpy as np
import pandas as pd
from streamlit_prophet.lib.evaluation.preparation import add_time_groupers
from streamlit_prophet.lib.utils.mapping import convert_into_nb_of_days, convert_into_nb_of_seconds


def MAPE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Computes Mean Absolute Percentage Error (MAPE). Must be multiplied by 100 to get percentage.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth target series.
    y_pred : pd.Series
        Prediction series.

    Returns
    -------
    float
        Mean Absolute Percentage Error (MAPE).
    """
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (y_true != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
        y_true, y_pred = y_true, y_pred
        mape = np.mean(np.abs((y_true - y_pred) / y_true)[mask])
        return 0 if np.isnan(mape) else float(mape)
    except:
        return 0


def SMAPE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Computes Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters
    ----------
    y_true : pd.Series
        Ground truth target series.
    y_pred : pd.Series
        Prediction series.

    Returns
    -------
    float
        Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (abs(y_true) + abs(y_pred) != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
        y_true, y_pred = y_true, y_pred
        nominator = np.abs(y_true - y_pred)
        denominator = np.abs(y_true) + np.abs(y_pred)
        smape = np.mean((2.0 * nominator / denominator)[mask])
        return 0 if np.isnan(smape) else float(smape)
    except:
        return 0


def MSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Computes Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : pd.Series
        Ground truth target series.
    y_pred : pd.Series
        Prediction series.

    Returns
    -------
    float
        Mean Squared Error (MSE).
    """
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        mse = ((y_true - y_pred) ** 2)[mask].mean()
        return 0 if np.isnan(mse) else float(mse)
    except:
        return 0


def RMSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Computes Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : pd.Series
        Ground truth target series.
    y_pred : pd.Series
        Prediction series.

    Returns
    -------
    float
        Root Mean Squared Error (RMSE).
    """
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    rmse = np.sqrt(MSE(y_true, y_pred))
    return float(rmse)


def MAE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Computes Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : pd.Series
        Ground truth target series.
    y_pred : pd.Series
        Prediction series.

    Returns
    -------
    float
        Mean Absolute Error (MAE).
    """
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        mae = abs(y_true - y_pred)[mask].mean()
        return 0 if np.isnan(mae) else float(mae)
    except:
        return 0


def get_perf_metrics(
    evaluation_df: pd.DataFrame,
    eval: Dict[Any, Any],
    dates: Dict[Any, Any],
    resampling: Dict[Any, Any],
    use_cv: bool,
    config: Dict[Any, Any],
) -> Tuple[pd.DataFrame, Dict[Any, Any]]:
    """Computes all metrics to gather them in a dataframe and a dictionary.

    Parameters
    ----------
    evaluation_df : pd.DataFrame
        Evaluation dataframe.
    eval : Dict
        Evaluation specifications.
    dates : Dict
        Dictionary containing all dates information.
    resampling : Dict
        Resampling specifications.
    use_cv : bool
        Whether or note cross-validation is used.
    config : Dict
        Lib configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Dataframe with all metrics at the desired granularity.
    dict
        Dictionary with all metrics at the desired granularity.
    """
    df = _preprocess_eval_df(evaluation_df, use_cv)
    metrics_df = _compute_metrics(df, eval)
    metrics_df, metrics_dict = _format_eval_results(
        metrics_df, dates, eval, resampling, use_cv, config
    )
    return metrics_df, metrics_dict


def _preprocess_eval_df(evaluation_df: pd.DataFrame, use_cv: bool) -> pd.DataFrame:
    """Preprocesses evaluation dataframe.

    Parameters
    ----------
    evaluation_df : pd.DataFrame
        Evaluation dataframe.
    use_cv : bool
        Whether or note cross-validation is used.

    Returns
    -------
    pd.DataFrame
        Preprocessed evaluation dataframe.
    """
    if use_cv:
        df = evaluation_df.copy()
    else:
        df = add_time_groupers(evaluation_df)
    return df


def _compute_metrics(df: pd.DataFrame, eval: Dict[Any, Any]) -> pd.DataFrame:
    """Computes all metrics and gather them in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation dataframe.
    eval : Dict
        Evaluation specifications.

    Returns
    -------
    pd.DataFrame
        Dataframe with all metrics at the desired granularity.
    """
    metrics = {"MAPE": MAPE, "SMAPE": SMAPE, "MSE": MSE, "RMSE": RMSE, "MAE": MAE}
    if eval["get_perf_on_agg_forecast"]:
        metrics_df = (
            df.groupby(eval["granularity"]).agg({"truth": "sum", "forecast": "sum"}).reset_index()
        )
        for m in eval["metrics"]:
            metrics_df[m] = metrics_df[["truth", "forecast"]].apply(
                lambda x: metrics[m](x.truth, x.forecast), axis=1
            )
    else:
        metrics_df = pd.DataFrame({eval["granularity"]: sorted(df[eval["granularity"]].unique())})
        for m in eval["metrics"]:
            metrics_df[m] = (
                df.groupby(eval["granularity"])[["truth", "forecast"]]
                .apply(lambda x: metrics[m](x.truth, x.forecast))
                .sort_index()
                .to_list()
            )
    return metrics_df


def _format_eval_results(
    metrics_df: pd.DataFrame,
    dates: Dict[Any, Any],
    eval: Dict[Any, Any],
    resampling: Dict[Any, Any],
    use_cv: bool,
    config: Dict[Any, Any],
) -> Tuple[pd.DataFrame, Dict[Any, Any]]:
    """Formats dataframe containing evaluation results and creates a dictionary containing the same information.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Dataframe with all metrics at the desired granularity.
    dates : Dict
        Dictionary containing all dates information.
    eval : Dict
        Evaluation specifications.
    resampling : Dict
        Resampling specifications.
    use_cv : bool
        Whether or note cross-validation is used.
    config : Dict
        Lib configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Formatted dataframe with all metrics at the desired granularity.
    dict
        Dictionary with all metrics at the desired granularity.
    """
    if use_cv:
        metrics_df = __format_metrics_df_cv(metrics_df, dates, eval, resampling)
        metrics_dict = {m: metrics_df[[eval["granularity"], m]] for m in eval["metrics"]}
        metrics_df = __add_avg_std_metrics(metrics_df, eval)
    else:
        metrics_dict = {m: metrics_df[[eval["granularity"], m]] for m in eval["metrics"]}
        metrics_df = metrics_df[[eval["granularity"]] + eval["metrics"]].set_index(
            [eval["granularity"]]
        )
    metrics_df = __format_metrics_values(metrics_df, eval, config)
    return metrics_df, metrics_dict


def __format_metrics_values(
    metrics_df: pd.DataFrame, eval: Dict[Any, Any], config: Dict[Any, Any]
) -> pd.DataFrame:
    """Formats metrics values with the right number of decimals.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Dataframe with all metrics at the desired granularity.
    eval : Dict
        Evaluation specifications.
    config : Dict
        Lib configuration dictionary containing information about the number of decimals to keep.

    Returns
    -------
    pd.DataFrame
        Dataframe with all metrics formatted with the right number of decimals.
    """
    mapping_format = {k: "{:,." + str(v) + "f}" for k, v in config["metrics"]["digits"].items()}
    mapping_round = config["metrics"]["digits"].copy()
    for col in eval["metrics"]:
        metrics_df[col] = metrics_df[col].map(
            lambda x: mapping_format[col].format(round(x, mapping_round[col]))
        )
    return metrics_df


def __format_metrics_df_cv(
    metrics_df: pd.DataFrame,
    dates: Dict[Any, Any],
    eval: Dict[Any, Any],
    resampling: Dict[Any, Any],
) -> pd.DataFrame:
    """Formats dataframe containing evaluation metrics, in case cross-validation is used.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Dataframe with all metrics for each cross-validation fold.
    dates : Dict
        Dictionary containing cross-validation dates information.
    eval : Dict
        Evaluation specifications.
    resampling : Dict
        Resampling specifications.

    Returns
    -------
    pd.DataFrame
        Formatted dataframe with all metrics displayed for each cross-validation fold.
    """
    metrics_df = metrics_df.rename(columns={"cutoff": "Valid Start"})
    freq = resampling["freq"][-1]
    horizon = dates["folds_horizon"]
    if freq in ["s", "H"]:
        metrics_df["Valid End"] = (
            metrics_df["Valid Start"]
            .map(lambda x: x + timedelta(seconds=convert_into_nb_of_seconds(freq, horizon)))
            .astype(str)
        )
    else:
        metrics_df["Valid End"] = (
            metrics_df["Valid Start"]
            .map(lambda x: x + timedelta(days=convert_into_nb_of_days(freq, horizon)))
            .astype(str)
        )
    metrics_df["Valid Start"] = metrics_df["Valid Start"].astype(str)
    metrics_df = metrics_df.sort_values("Valid Start", ascending=False).reset_index(drop=True)
    metrics_df[eval["granularity"]] = [f"Fold {i}" for i in range(1, len(metrics_df) + 1)]
    return metrics_df


def __add_avg_std_metrics(metrics_df: pd.DataFrame, eval: Dict[Any, Any]) -> pd.DataFrame:
    """Adds rows for average and standard-deviation over each fold to dataframe containing evaluation metrics.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Dataframe with all metrics for each cross-validation fold.
    eval : Dict
        Evaluation specifications.

    Returns
    -------
    pd.DataFrame
        Formatted dataframe with two more rows (for average and standard-deviation of each metrics).
    """
    cols_index = [eval["granularity"], "Valid Start", "Valid End"]
    metrics_df = metrics_df[cols_index + eval["metrics"]].set_index(cols_index)
    metrics_df.loc[("Avg", "", "Average")] = metrics_df.mean(axis=0)
    metrics_df.loc[("Std", "", "+/-")] = metrics_df.std(axis=0)
    metrics_df = metrics_df.reset_index().set_index(eval["granularity"])
    return metrics_df
