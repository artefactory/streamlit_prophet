import pytest
from streamlit_prophet.lib.inputs.dataprep import _autodetect_dimensions, _autodetect_freq
from tests.samples.df import df_test


@pytest.mark.parametrize(
    "df, expected",
    [
        (df_test[0], []),
        (df_test[1], [0, 1]),
        (df_test[2], []),
        (df_test[13], [4, 5, 6]),
    ],
)
def test_autodetect_dimensions(df, expected):
    # The right dimension columns have been detected
    assert _autodetect_dimensions(df.copy()) == expected


@pytest.mark.parametrize(
    "df, expected",
    [
        (df_test[8], "D"),
        (df_test[10], "1Y"),
        (df_test[11], "1H"),
    ],
)
def test_autodetect_freq(df, expected):
    # The right frequency has been detected
    assert _autodetect_freq(df.copy()) == expected
