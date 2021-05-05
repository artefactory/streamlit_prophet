"""
Functions directly taken from Facebook Prophet repo, v0.7:
https://github.com/facebook/prophet/blob/master/python/prophet/diagnostics.py
Necessary only if using fbprophet < 0.7
"""

import concurrent.futures

import pandas as pd
from fbprophet.diagnostics import generate_cutoffs, logger, prophet_copy
from tqdm.auto import tqdm


def cross_validation(
    model, horizon, period=None, initial=None, parallel=None, cutoffs=None, disable_tqdm=False
):
    """Cross-Validation for time series.
    Computes forecasts from historical cutoff points, which user can input.
    If not provided, begins from (end - horizon) and works backwards, making
    cutoffs with a spacing of period until initial is reached.
    When period is equal to the time interval of the data, this is the
    technique described in https://robjhyndman.com/hyndsight/tscv/ .
    Parameters
    ----------
    model: Prophet class object. Fitted Prophet model.
    horizon: string with pd.Timedelta compatible style, e.g., '5 days',
        '3 hours', '10 seconds'.
    period: string with pd.Timedelta compatible style. Simulated forecast will
        be done at every this period. If not provided, 0.5 * horizon is used.
    initial: string with pd.Timedelta compatible style. The first training
        period will include at least this much data. If not provided,
        3 * horizon is used.
    cutoffs: list of pd.Timestamp specifying cutoffs to be used during
        cross validation. If not provided, they are generated as described
        above.
    parallel : {None, 'processes', 'threads', 'dask', object}
    disable_tqdm: if True it disables the progress bar that would otherwise show up when parallel=None
        How to parallelize the forecast computation. By default no parallelism
        is used.
        * None : No parallelism.
        * 'processes' : Parallelize with concurrent.futures.ProcessPoolExectuor.
        * 'threads' : Parallelize with concurrent.futures.ThreadPoolExecutor.
            Note that some operations currently hold Python's Global Interpreter
            Lock, so parallelizing with threads may be slower than training
            sequentially.
        * 'dask': Parallelize with Dask.
           This requires that a dask.distributed Client be created.
        * object : Any instance with a `.map` method. This method will
          be called with :func:`single_cutoff_forecast` and a sequence of
          iterables where each element is the tuple of arguments to pass to
          :func:`single_cutoff_forecast`
          .. code-block::
             class MyBackend:
                 def map(self, func, *iterables):
                     results = [
                        func(*args)
                        for args in zip(*iterables)
                     ]
                     return results
    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """

    df = model.history.copy().reset_index(drop=True)
    horizon = pd.Timedelta(horizon)

    predict_columns = ["ds", "yhat"]
    if model.uncertainty_samples:
        predict_columns.extend(["yhat_lower", "yhat_upper"])

    # Identify largest seasonality period
    period_max = 0.0
    for s in model.seasonalities.values():
        period_max = max(period_max, s["period"])
    seasonality_dt = pd.Timedelta(str(period_max) + " days")

    if cutoffs is None:
        # Set period
        period = 0.5 * horizon if period is None else pd.Timedelta(period)

        # Set initial
        initial = max(3 * horizon, seasonality_dt) if initial is None else pd.Timedelta(initial)

        # Compute Cutoffs
        cutoffs = generate_cutoffs(df, horizon, initial, period)
    else:
        # add validation of the cutoff to make sure that the min cutoff is strictly greater than the min date in the history
        if min(cutoffs) <= df["ds"].min():
            raise ValueError(
                "Minimum cutoff value is not strictly greater than min date in history"
            )
        # max value of cutoffs is <= (end date minus horizon)
        end_date_minus_horizon = df["ds"].max() - horizon
        if max(cutoffs) > end_date_minus_horizon:
            raise ValueError(
                "Maximum cutoff value is greater than end date minus horizon, no value for cross-validation remaining"
            )
        initial = cutoffs[0] - df["ds"].min()

    # Check if the initial window
    # (that is, the amount of time between the start of the history and the first cutoff)
    # is less than the maximum seasonality period
    if initial < seasonality_dt:
        msg = f"Seasonality has period of {period_max} days "
        msg += "which is larger than initial window. "
        msg += "Consider increasing initial."
        logger.warning(msg)

    if parallel:
        valid = {"threads", "processes", "dask"}

        if parallel == "threads":
            pool = concurrent.futures.ThreadPoolExecutor()
        elif parallel == "processes":
            pool = concurrent.futures.ProcessPoolExecutor()
        elif parallel == "dask":
            try:
                from dask.distributed import get_client
            except ImportError as e:
                raise ImportError("parallel='dask' requies the optional " "dependency dask.") from e
            pool = get_client()
            # delay df and model to avoid large objects in task graph.
            df, model = pool.scatter([df, model])
        elif hasattr(parallel, "map"):
            pool = parallel
        else:
            msg = "'parallel' should be one of {} for an instance with a " "'map' method".format(
                ", ".join(valid)
            )
            raise ValueError(msg)

        iterables = ((df, model, cutoff, horizon, predict_columns) for cutoff in cutoffs)
        iterables = zip(*iterables)

        logger.info("Applying in parallel with %s", pool)
        predicts = pool.map(single_cutoff_forecast, *iterables)
        if parallel == "dask":
            # convert Futures to DataFrames
            predicts = pool.gather(predicts)

    else:
        predicts = [
            single_cutoff_forecast(df, model, cutoff, horizon, predict_columns)
            for cutoff in (tqdm(cutoffs) if not disable_tqdm else cutoffs)
        ]

    # Combine all predicted pd.DataFrame into one pd.DataFrame
    return pd.concat(predicts, axis=0).reset_index(drop=True)


def single_cutoff_forecast(df, model, cutoff, horizon, predict_columns):
    """Forecast for single cutoff. Used in cross validation function
    when evaluating for multiple cutoffs either sequentially or in parallel .
    Parameters
    ----------
    df: pd.DataFrame.
        DataFrame with history to be used for single
        cutoff forecast.
    model: Prophet model object.
    cutoff: pd.Timestamp cutoff date.
        Simulated Forecast will start from this date.
    horizon: pd.Timedelta forecast horizon.
    predict_columns: List of strings e.g. ['ds', 'yhat'].
        Columns with date and forecast to be returned in output.
    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """

    # Generate new object with copying fitting options
    m = prophet_copy(model, cutoff)
    # Train model
    history_c = df[df["ds"] <= cutoff]
    if history_c.shape[0] < 2:
        raise Exception("Less than two datapoints before cutoff. " "Increase initial window.")
    m.fit(history_c, **model.fit_kwargs)
    # Calculate yhat
    index_predicted = (df["ds"] > cutoff) & (df["ds"] <= cutoff + horizon)
    # Get the columns for the future dataframe
    columns = ["ds"]
    if m.growth == "logistic":
        columns.append("cap")
        if m.logistic_floor:
            columns.append("floor")
    columns.extend(m.extra_regressors.keys())
    columns.extend(
        [
            props["condition_name"]
            for props in m.seasonalities.values()
            if props["condition_name"] is not None
        ]
    )
    yhat = m.predict(df[index_predicted][columns])
    # Merge yhat(predicts), y(df, original data) and cutoff

    return pd.concat(
        [
            yhat[predict_columns],
            df[index_predicted][["y"]].reset_index(drop=True),
            pd.DataFrame({"cutoff": [cutoff] * len(yhat)}),
        ],
        axis=1,
    )
