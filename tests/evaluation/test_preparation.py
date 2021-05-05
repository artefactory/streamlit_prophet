import pytest
from streamlit_prophet.lib.evaluation.preparation import add_time_groupers
from tests.samples.df import df_test


@pytest.mark.parametrize(
    "df",
    [df_test[8], df_test[10], df_test[11]],
)
def test_add_time_groupers(df):
    output = add_time_groupers(df)
    assert output.shape[0] == df.shape[0]
    assert output.shape[1] == df.shape[1] + 6
