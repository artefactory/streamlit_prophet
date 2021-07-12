from typing import Any, Dict

import pandas as pd


def get_evaluation_df(
    datasets: Dict[Any, Any],
    forecasts: Dict[Any, Any],
    dates: Dict[Any, Any],
    eval: Dict[Any, Any],
    use_cv: bool,
) -> pd.DataFrame:
    """Generates a dataframe that will be used for evaluation.

    Parameters
    ----------
    datasets : Dict
        Dictionary containing evaluation dataframe.
    forecasts : Dict
        Dictionary where all forecasts are stored.
    dates : Dict
        Dictionary containing all dates information.
    eval : Dict
        Evaluation specifications.
    use_cv : bool
        Whether or not cross-validation is used.

    Returns
    -------
    pd.DataFrame
        Evaluation dataframe.
    """
    if use_cv:
        evaluation_df = forecasts["cv"].rename(columns={"y": "truth", "yhat": "forecast"})
        mapping = {
            cutoff: f"Fold {i + 1}"
            for i, cutoff in enumerate(sorted(evaluation_df["cutoff"].unique(), reverse=True))
        }
        evaluation_df["Fold"] = evaluation_df["cutoff"].map(mapping)
        evaluation_df = evaluation_df.sort_values("ds")
    else:
        evaluation_df = pd.DataFrame()
        if eval["set"] == "Validation":
            evaluation_df["ds"] = datasets["val"].ds.copy()
            evaluation_df["truth"] = list(datasets["val"].y)
            evaluation_df["forecast"] = list(
                forecasts["eval"]
                .query(f'ds >= "{dates["val_start_date"]}" & ' f'ds <= "{dates["val_end_date"]}"')
                .yhat
            )
        elif eval["set"] == "Training":
            evaluation_df["ds"] = datasets["train"].ds.copy()
            evaluation_df["truth"] = list(datasets["train"].y)
            evaluation_df["forecast"] = list(
                forecasts["eval"]
                .query(
                    f'ds >= "{dates["train_start_date"]}" & ' f'ds <= "{dates["train_end_date"]}"'
                )
                .yhat
            )
    return evaluation_df


def add_time_groupers(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Adds columns with time information (day, week, quarter, year) to evaluation dataframe.

    Parameters
    ----------
    evaluation_df : pd.DataFrame
        Dictionary containing evaluation dataframe.

    Returns
    -------
    pd.DataFrame
        Evaluation dataframe with additional time information columns.
    """
    df = evaluation_df.copy()
    df["Global"] = "Global"
    df["Daily"] = df["ds"].astype(str).map(lambda x: x[0:10])
    df["Day of Week"] = (
        df["ds"].dt.dayofweek.map(lambda x: x + 1).astype(str) + ". " + df["ds"].dt.day_name()
    )
    df["Weekly"] = (
        df["ds"].dt.year.astype(str)
        + " - W"
        + df["ds"].dt.isocalendar().week.astype(str).map(lambda x: "0" + x if len(x) < 2 else x)
    )
    df["Monthly"] = (
        df["ds"].dt.year.astype(str)
        + " - M"
        + df["ds"].dt.month.astype(str).map(lambda x: "0" + x if len(x) < 2 else x)
    )
    df["Quarterly"] = df["ds"].dt.year.astype(str) + " - Q" + df["ds"].dt.quarter.astype(str)
    df["Yearly"] = df["ds"].dt.year.astype(str)
    return df
