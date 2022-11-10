import itertools

import pytest
from streamlit_prophet.lib.dataprep.format import (
    filter_and_aggregate_df,
    format_date_and_target,
    remove_empty_cols,
    resample_df,
)
from streamlit_prophet.lib.utils.load import load_config
from tests.samples.df import df_test
from tests.samples.dict import make_dimensions_test, make_resampling_test

config, _, _ = load_config(
    "config_streamlit.toml", "config_instructions.toml", "config_readme.toml"
)


@pytest.mark.parametrize(
    "df, expected0, expected1",
    [
        (df_test[0], df_test[0], []),
        (df_test[1], df_test[1][[0, 1, 3]], [2]),
        (df_test[2], df_test[2][[1]], [0, 2]),
    ],
)
def test_remove_empty_cols(df, expected0, expected1):
    # Check if the right columns are staying in the output dataframe
    assert remove_empty_cols(df.copy())[0].equals(expected0)
    # Check if the right columns have been removed
    assert remove_empty_cols(df.copy())[1] == expected1


# Temporarily deactivate this test as script_runner is deprecated
# @pytest.mark.parametrize(
#     "df, date_col",
#    [
#        (df_test[0], ""),
#        (df_test[1], 0),
#        (df_test[1], 3),
#        (df_test[2], 0),
#        (df_test[2], 1),
#    ],
# )
# def test_format_date(df, date_col):
#    # Streamlit should stop and display an error message
#    with pytest.raises(st.script_runner.StopException):
#        load_options = {"date_format": config["dataprep"]["date_format"]}
#        _format_date(df.copy(), date_col, load_options, config)

# Temporarily deactivate this test as script_runner is deprecated
# @pytest.mark.parametrize(
#    "df, target_col",
#    list(
#        itertools.product(
#            [df_test[3], df_test[4], df_test[5], df_test[6], df_test[7]], ["y", "abc"]
#        )
#    ),
# )
# def test_format_target(df, target_col):
#    # Streamlit should stop and display an error message
#    with pytest.raises(st.script_runner.StopException):
#        _format_target(df.copy(), target_col, config)


@pytest.mark.parametrize(
    "df, date_col, target_col",
    list(
        itertools.product(
            [df_test[8], df_test[9], df_test[10], df_test[11], df_test[12]], ["ds"], ["y"]
        )
    ),
)
def test_format_date_and_target(df, date_col, target_col):
    load_options = {"date_format": config["dataprep"]["date_format"]}
    output = format_date_and_target(df.copy(), date_col, target_col, config, load_options)
    # Date column should have the same number of unique values in input and output dataframes
    assert output["ds"].nunique() == df[date_col].nunique()
    # Target column should have the same number of unique values in input and output dataframes
    assert output["y"].nunique() == df[target_col].nunique()
    # Target maximum value should be the same in input and output dataframes
    assert output["y"].max() == df[target_col].max()
    # Target minimum value should be the same in input and output dataframes
    assert output["y"].min() == df[target_col].min()
    # Target average value should be the same in input and output dataframes
    assert output["y"].mean() == df[target_col].mean()
    # Date column should be of type datetime in output dataframe
    assert output.dtypes["ds"].name == "datetime64[ns]"
    # Target column should be of type float in output dataframe
    assert output.dtypes["y"].name == "float64"
    # Input and output dataframes should have the same shape
    assert output.shape == df.shape


@pytest.mark.parametrize(
    "df, expected_dim, expected_drop",
    list(
        itertools.product(
            [df_test[14]("D"), df_test[14]("W"), df_test[14]("H")],
            [[4, 5, 6, 7, 8, 9, 10]],
            [[2, 11]],
        )
    ),
)
def test_filter_and_aggregate_df(df, expected_dim, expected_drop):
    dimensions = make_dimensions_test(df, frac=1)
    output1, output2 = filter_and_aggregate_df(
        df.copy(), dimensions=dimensions, config=config, date_col="", target_col=""
    )
    # The expected dimensions should have been detected automatically in input dataframe
    assert sorted(set(dimensions.keys()) - {"agg"}) == expected_dim
    # The expected columns should have been dropped automatically from input dataframe
    assert output2 == expected_drop
    # Output dataframe should have the same number or less rows than input dataframe
    assert len(output1) <= len(df)


@pytest.mark.parametrize(
    "origin_dim, new_dim, agg",
    [
        ("H", "D", "Mean"),
        ("D", "W", "Sum"),
        ("W", "M", "Max"),
        ("M", "Q", "Min"),
        ("D", "M", "Mean"),
        ("H", "Y", "Sum"),
    ],
)
def test_resample_df(origin_dim, new_dim, agg):
    df = df_test[14](origin_dim)[["ds", "y", 0, 1, 2, 3]].copy()
    resampling = make_resampling_test(freq=new_dim, agg=agg)
    output = resample_df(df, resampling=resampling)
    # Output dataframe should have less rows than input dataframe
    assert output.shape[0] < df.shape[0]
    # Output dataframe should have the same columns as input dataframe
    assert set(output.columns) == set(df.columns)
