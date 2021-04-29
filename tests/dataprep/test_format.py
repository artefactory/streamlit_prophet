import pytest
import streamlit as st
from lib.dataprep.format import remove_empty_cols, _format_date
from tests.samples.df import dataframes


@pytest.mark.parametrize(
    "df, expected0, expected1",
    [
        (dataframes[0], dataframes[0], []),
        (dataframes[1], dataframes[1][[0, 1]], [2]),
        (dataframes[2], dataframes[2][[1]], [0, 2]),
        (dataframes['M5'], dataframes['M5'], []),
    ],
)
def test_remove_empty_cols(df, expected0, expected1):
    assert remove_empty_cols(df)[0].equals(expected0)
    assert remove_empty_cols(df)[1] == expected1


@pytest.mark.parametrize(
    "df, date_col",
    [
        (dataframes['M5'], 'abc'),
        (dataframes['M5'], 'sales'),
        (dataframes[0], ''),
        (dataframes[1], 0),
        (dataframes[2], 0),
        (dataframes[2], 1),
    ],
)
def test_format_date_stop(df, date_col):
    with pytest.raises(st.script_runner.StopException):
        _format_date(df, date_col)


@pytest.mark.parametrize(
    "df, date_col",
    [
        (dataframes['M5'], 'date'),
        (dataframes[3], 'ds'),
        (dataframes[4], 'ds'),
        (dataframes[5], 'ds'),
        (dataframes[6], 'ds'),
        (dataframes[7], 'ds'),
    ],
)
def test_format_date(df, date_col):
    assert _format_date(df, date_col)[date_col].nunique() == df[date_col].nunique()
