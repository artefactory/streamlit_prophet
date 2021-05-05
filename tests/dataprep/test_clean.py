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
    assert output.shape[1] == df.shape[1]
    assert (
        output.ds.dt.dayofweek.nunique() + len(cleaning["del_days"]) == df.ds.dt.dayofweek.nunique()
    )
    if cleaning["del_negative"]:
        assert output.y.min() >= 0
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
    assert output.y.mean() > expected_min
    assert output.y.mean() < expected_max
    assert output.shape == df.shape
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
    assert output.shape[1] == df.shape[1]
    assert (
        output.ds.dt.dayofweek.nunique() + len(cleaning["del_days"]) == df.ds.dt.dayofweek.nunique()
    )
