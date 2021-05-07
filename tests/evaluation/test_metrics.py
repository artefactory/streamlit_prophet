import itertools

import numpy as np
import pandas as pd
import pytest
from streamlit_prophet.lib.evaluation.metrics import MAE, MAPE, MSE, RMSE, SMAPE, _compute_metrics
from streamlit_prophet.lib.evaluation.preparation import add_time_groupers
from tests.samples.df import df_test
from tests.samples.dict import make_eval_test


@pytest.mark.parametrize(
    "y_true, y_pred, expected_min, expected_max",
    [
        (pd.Series(range(1, 11)), pd.Series(range(0, 10)), 0.29, 0.30),
        (pd.Series(range(0, 10)), pd.Series(range(1, 11)), 0.31, 0.32),
        (pd.Series(), pd.Series(range(1, 11)), 0, 0),
        (pd.Series(range(1, 11)), pd.Series(), 0, 0),
        (pd.Series(), pd.Series(), 0, 0),
        (pd.Series([1, np.nan, np.nan]), pd.Series([1, 1, np.nan]), 0, 0),
        (pd.Series([1, 2, np.nan]), pd.Series([1, 1, np.nan]), 0.25, 0.25),
    ],
)
def test_MAPE(y_true, y_pred, expected_min, expected_max):
    output = MAPE(y_true, y_pred)
    # MAPE should have the expected value
    assert output >= expected_min
    assert output <= expected_max


@pytest.mark.parametrize(
    "y_true, y_pred, expected_min, expected_max",
    [
        (pd.Series(range(1, 11)), pd.Series(range(0, 10)), 0.42, 0.43),
        (pd.Series(range(0, 10)), pd.Series(range(1, 11)), 0.42, 0.43),
        (pd.Series(), pd.Series(range(1, 11)), 0, 0),
        (pd.Series(range(1, 11)), pd.Series(), 0, 0),
        (pd.Series(), pd.Series(), 0, 0),
        (pd.Series([1, np.nan, np.nan]), pd.Series([1, 1, np.nan]), 0, 0),
        (pd.Series([1, 2, np.nan]), pd.Series([1, 1, np.nan]), 0.33, 0.34),
    ],
)
def test_SMAPE(y_true, y_pred, expected_min, expected_max):
    output = SMAPE(y_true, y_pred)
    # SMAPE should have the expected value
    assert output >= expected_min
    assert output <= expected_max


@pytest.mark.parametrize(
    "y_true, y_pred, expected_min, expected_max",
    [
        (pd.Series(range(1, 11)), pd.Series(range(0, 10)), 1, 1),
        (pd.Series(range(0, 10)), pd.Series(range(1, 11)), 1, 1),
        (pd.Series(), pd.Series(range(1, 11)), 0, 0),
        (pd.Series(range(1, 11)), pd.Series(), 0, 0),
        (pd.Series(), pd.Series(), 0, 0),
        (pd.Series([1, np.nan, np.nan]), pd.Series([1, 1, np.nan]), 0, 0),
        (pd.Series([1, 2, np.nan]), pd.Series([1, 1, np.nan]), 0.5, 0.5),
    ],
)
def test_MSE(y_true, y_pred, expected_min, expected_max):
    output = MSE(y_true, y_pred)
    # MSE should have the expected value
    assert output >= expected_min
    assert output <= expected_max


@pytest.mark.parametrize(
    "y_true, y_pred, expected_min, expected_max",
    [
        (pd.Series(range(1, 11)), pd.Series(range(0, 10)), 1, 1),
        (pd.Series(range(0, 10)), pd.Series(range(1, 11)), 1, 1),
        (pd.Series(), pd.Series(range(1, 11)), 0, 0),
        (pd.Series(range(1, 11)), pd.Series(), 0, 0),
        (pd.Series(), pd.Series(), 0, 0),
        (pd.Series([1, np.nan, np.nan]), pd.Series([1, 1, np.nan]), 0, 0),
        (pd.Series([1, 2, np.nan]), pd.Series([1, 1, np.nan]), 0.70, 0.71),
    ],
)
def test_RMSE(y_true, y_pred, expected_min, expected_max):
    output = RMSE(y_true, y_pred)
    # RMSE should have the expected value
    assert output >= expected_min
    assert output <= expected_max


@pytest.mark.parametrize(
    "y_true, y_pred, expected_min, expected_max",
    [
        (pd.Series(range(1, 11)), pd.Series(range(0, 10)), 1, 1),
        (pd.Series(range(0, 10)), pd.Series(range(1, 11)), 1, 1),
        (pd.Series(), pd.Series(range(1, 11)), 0, 0),
        (pd.Series(range(1, 11)), pd.Series(), 0, 0),
        (pd.Series(), pd.Series(), 0, 0),
        (pd.Series([1, np.nan, np.nan]), pd.Series([1, 1, np.nan]), 0, 0),
        (pd.Series([1, 2, np.nan]), pd.Series([1, 1, np.nan]), 0.5, 0.5),
    ],
)
def test_MAE(y_true, y_pred, expected_min, expected_max):
    output = MAE(y_true, y_pred)
    # MAE should have the expected value
    assert output >= expected_min
    assert output <= expected_max


@pytest.mark.parametrize(
    "df, eval",
    list(
        itertools.product(
            [df_test[17], df_test[18], df_test[19]],
            [
                make_eval_test(),
                make_eval_test(granularity="Weekly", get_perf_on_agg_forecast=True),
                make_eval_test(granularity="Monthly"),
                make_eval_test(granularity="Global", get_perf_on_agg_forecast=True),
            ],
        )
    ),
)
def test_compute_metrics(df, eval):
    df = add_time_groupers(df)
    output = _compute_metrics(df, eval)
    expected_cols = ["MAPE", "SMAPE", "RMSE", "MSE", "MAE", eval["granularity"]]
    # There shouldn't be any NaN values in metrics dataframe
    assert output[expected_cols].isnull().sum().sum() == 0
    # Metrics dataframe should have the expected columns
    assert sorted(output.columns) == sorted(
        expected_cols + ["forecast", "truth"] if eval["get_perf_on_agg_forecast"] else expected_cols
    )
