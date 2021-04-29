import pytest
import streamlit as st
from lib.utils.load import load_config
from lib.dataprep.format import (remove_empty_cols,
                                 _format_date,
                                 _format_target,
                                 format_date_and_target,
                                 filter_and_aggregate_df,
                                 resample_df
                                 )
from tests.samples.df import df_test
from tests.samples.dict import make_dimensions_test, make_resampling_test

config, _ = load_config('config_streamlit.toml', 'config_readme.toml')


@pytest.mark.parametrize(
    "df, expected0, expected1",
    [
        (df_test[0], df_test[0], []),
        (df_test[1], df_test[1][[0, 1]], [2]),
        (df_test[2], df_test[2][[1]], [0, 2]),
        (df_test['M5'], df_test['M5'], []),
    ],
)
def test_remove_empty_cols(df, expected0, expected1):
    assert remove_empty_cols(df.copy())[0].equals(expected0)
    assert remove_empty_cols(df.copy())[1] == expected1


@pytest.mark.parametrize(
    "df, date_col",
    [
        (df_test['M5'], 'abc'),
        (df_test['M5'], 'sales'),
        (df_test[0], ''),
        (df_test[1], 0),
        (df_test[2], 0),
        (df_test[2], 1),
    ],
)
def test_format_date_stop(df, date_col):
    with pytest.raises(st.script_runner.StopException):
        _format_date(df.copy(), date_col)


@pytest.mark.parametrize(
    "df, target_col",
    [
        (df_test['M5'], 'abc'),
        (df_test['M5'], 'date'),
        (df_test[3], 'y'),
        (df_test[4], 'y'),
        (df_test[5], 'y'),
        (df_test[6], 'y'),
        (df_test[7], 'y'),
    ],
)
def test_format_target_stop(df, target_col):
    with pytest.raises(st.script_runner.StopException):
        _format_target(df.copy(), target_col)


@pytest.mark.parametrize(
    "df, date_col, target_col",
    [
        (df_test['M5'], config['datasets']['M5']['date'], config['datasets']['M5']['target']),
        (df_test['SAV'], config['datasets']['SAV']['date'], config['datasets']['SAV']['target']),
        (df_test['Weather'], config['datasets']['Weather']['date'], config['datasets']['Weather']['target']),
        (df_test[8], 'ds', 'y'),
        (df_test[9], 'ds', 'y'),
        (df_test[10], 'ds', 'y'),
        (df_test[11], 'ds', 'y'),
        (df_test[12], 'ds', 'y'),
    ],
)
def test_format_date_and_target(df, date_col, target_col):
    output = format_date_and_target(df.copy(), date_col, target_col)
    assert output['ds'].nunique() == df[date_col].nunique()
    assert output['y'].nunique() == df[target_col].nunique()
    assert output.dtypes['ds'].name == 'datetime64[ns]'
    assert output.dtypes['y'].name == 'float64'
    assert output.shape == df.shape


@pytest.mark.parametrize(
    "df, expected_dim, expected_drop",
    [
        (df_test[14]('D'), [4, 5, 6, 7, 8, 9, 10], [2, 11]),
        (df_test[14]('W'), [4, 5, 6, 7, 8, 9, 10], [2, 11]),
        (df_test[14]('H'), [4, 5, 6, 7, 8, 9, 10], [2, 11]),
    ],
)
def test_filter_and_aggregate_df(df, expected_dim, expected_drop):
    dimensions = make_dimensions_test(df, frac=1)
    output1, output2 = filter_and_aggregate_df(df.copy(), dimensions=dimensions, date_col='', target_col='')
    assert sorted(set(dimensions.keys()) - set(['agg'])) == expected_dim
    assert output2 == expected_drop
    assert len(output1) <= len(df)


@pytest.mark.parametrize(
    "origin_dim, new_dim, agg",
    [
        ('H', 'D', 'Mean'),
        ('D', 'W', 'Sum'),
        ('W', 'M', 'Max'),
        ('M', 'Q', 'Min'),
        ('D', 'M', 'Mean'),
        ('H', 'Y', 'Sum'),
    ],
)
def test_resample_df(origin_dim, new_dim, agg):
    df = df_test[14](origin_dim)[['ds', 'y', 0, 1, 2, 3]].copy()
    resampling = make_resampling_test(freq=new_dim, agg=agg)
    output = resample_df(df, resampling=resampling)
    assert output.shape[0] < df.shape[0]
    assert set(output.columns) == set(df.columns)
