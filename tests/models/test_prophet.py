import pytest
from streamlit_prophet.lib.dataprep.split import get_train_set, get_train_val_sets
from streamlit_prophet.lib.models.prophet import forecast_workflow
from streamlit_prophet.lib.utils.load import load_config
from tests.samples.df import df_test
from tests.samples.dict import (
    make_cleaning_test,
    make_dates_test,
    make_dimensions_test,
    make_params_test,
    make_resampling_test,
)

config, _, _ = load_config(
    "config_streamlit.toml", "config_instructions.toml", "config_readme.toml"
)


@pytest.mark.parametrize(
    "use_cv, make_future_forecast, evaluate",
    [
        (True, False, True),
        (False, True, False),
        (True, True, True),
    ],
)
def test_forecast_workflow(use_cv, make_future_forecast, evaluate):
    df = df_test[20]
    params = make_params_test(
        regressors={
            col: {"prior_scale": 10, "mode": "additive"} for col in set(df.columns) - {"ds", "y"}
        }
    )
    dates = make_dates_test()
    cleaning = make_cleaning_test()
    resampling = make_resampling_test()
    dimensions = make_dimensions_test(df, frac=1)
    load_options = {"date_format": "%Y-%m-%d"}
    date_col, target_col = "ds", "y"
    datasets = (
        get_train_set(df, dates, dict())
        if use_cv
        else get_train_val_sets(df, dates, config, dict())
    )
    datasets, models, forecasts = forecast_workflow(
        config,
        use_cv,
        make_future_forecast,
        evaluate,
        cleaning,
        resampling,
        params,
        dates,
        datasets,
        df,
        date_col,
        target_col,
        dimensions,
        load_options,
    )
    # All dataframes in forecasts dictionary have at least a 'ds' column and a 'yhat' column
    assert all([len({"ds", "yhat"}.intersection(set(x.columns))) == 2 for x in forecasts.values()])
    # Models dictionary contains 2 models if a forecast has been made on future dates and 1 otherwise
    assert len(models) == make_future_forecast + evaluate
    if evaluate:
        # Training dataframe has at least 2 distinct dates
        assert datasets["train"].ds.nunique() > 1
        if use_cv:
            # Number of distinct dates in CV dataframe = number of cutoffs * the horizon length
            assert forecasts["cv"].ds.nunique() == len(dates["cutoffs"]) * dates["folds_horizon"]
            # Number of distinct dates in CV dataframe concatenated with past predictions
            # is equal to the number of distinct dates in training dataframe
            assert forecasts["cv_with_hist"].ds.nunique() == datasets["train"].ds.nunique()
        else:
            # Number of distinct dates in eval dataframe = number of distinct dates in train and val dataframes
            assert (
                datasets["eval"].ds.nunique()
                == datasets["train"].ds.nunique() + datasets["val"].ds.nunique()
            )
            # Number of distinct dates in eval dataframe = number of distinct dates in eval forecast dataframe
            assert forecasts["eval"].ds.nunique() == datasets["eval"].ds.nunique()
    if make_future_forecast:
        # Full dataframe has the same number of distinct dates as the input dataframe
        assert datasets["full"].ds.nunique() == df.ds.nunique()
        # Number of distinct dates in future dataframe is at least 1
        assert datasets["future"].ds.nunique() > 0
        # Number of distinct dates in future dataframe = number of distinct dates in future forecast dataframe
        assert forecasts["future"].ds.nunique() == datasets["future"].ds.nunique()
