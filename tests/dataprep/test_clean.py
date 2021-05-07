import itertools

import pytest
from streamlit_prophet.lib.dataprep.clean import _log_transform, _remove_rows, clean_future_df
from tests.samples.df import df_test
from tests.samples.dict import make_cleaning_test


@pytest.mark.parametrize(
    "df, cleaning",
    list(
        itertools.product(
            [df_test[8], df_test[11], df_test[12]],
            [
                make_cleaning_test(),
                make_cleaning_test(del_days=[6]),
                make_cleaning_test(del_days=[0, 1, 2, 3, 4, 5]),
                make_cleaning_test(del_zeros=False),
                make_cleaning_test(del_negative=False),
            ],
        )
    ),
)
def test_remove_rows(df, cleaning):
    output = _remove_rows(df.copy(), cleaning)
    # Output dataframe should have the same number of columns than input dataframe
    assert output.shape[1] == df.shape[1]
    # Number of distinct days of the week in output dataframe + Number of days removed
    # should be equal to number of distinct days of the week in input dataframe
    assert (
        output.ds.dt.dayofweek.nunique() + len(cleaning["del_days"]) == df.ds.dt.dayofweek.nunique()
    )
    # Minimum value for y in output dataframe should be positive if del_negative option was selected
    if cleaning["del_negative"]:
        assert output.y.min() >= 0
    # Minimum abs value for y in output dataframe should be strictly positive if del_zeros option was selected
    if cleaning["del_zeros"]:
        assert abs(output.y).min() > 0


@pytest.mark.parametrize(
    "df, cleaning, expected_min, expected_max",
    [
        (df_test[15], make_cleaning_test(log_transform=True), 0.69, 0.7),
        (df_test[16], make_cleaning_test(log_transform=True), 1.09, 1.1),
        (df_test[12].loc[df_test[12]["y"] > 0.1], make_cleaning_test(log_transform=True), -10, 10),
    ],
)
def test_log_transform(df, cleaning, expected_min, expected_max):
    output = _log_transform(df.copy(), cleaning)
    # y value should be in the expected range in output dataframe
    assert output.y.mean() > expected_min
    assert output.y.mean() < expected_max
    # Output dataframe should have the same shape as input dataframe
    assert output.shape == df.shape
    # Number of distinct y values should be the same in input and output dataframes
    assert output.y.nunique() == df.y.nunique()


@pytest.mark.parametrize(
    "df, cleaning",
    list(
        itertools.product(
            [df_test[8], df_test[11], df_test[12]],
            [
                make_cleaning_test(),
                make_cleaning_test(del_days=[6]),
                make_cleaning_test(del_days=[0, 1, 2, 3, 4, 5]),
            ],
        )
    ),
)
def test_clean_future_df(df, cleaning):
    output = clean_future_df(df.copy(), cleaning)
    # Output dataframe should have the same number of columns than input dataframe
    assert output.shape[1] == df.shape[1]
    # Number of distinct days of the week in output dataframe + Number of days removed
    # should be equal to number of distinct days of the week in input dataframe
    assert (
        output.ds.dt.dayofweek.nunique() + len(cleaning["del_days"]) == df.ds.dt.dayofweek.nunique()
    )
