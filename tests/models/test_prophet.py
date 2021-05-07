import pytest
from streamlit_prophet.lib.dataprep.split import get_train_set, get_train_val_sets
from streamlit_prophet.lib.models.prophet import forecast_workflow
from streamlit_prophet.lib.utils.load import load_config
from tests.samples.df import df_test
from tests.samples.dict import (
    make_cleaning_test,
    make_dates_test,
    make_params_test,
    make_resampling_test,
)

config, _ = load_config("config_streamlit.toml", "config_readme.toml")


@pytest.mark.parametrize(
    "use_cv, make_future_forecast",
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_forecast_workflow(use_cv, make_future_forecast):
    df = df_test[20]
    params = make_params_test(
        regressors={
            col: {"prior_scale": 10, "mode": "additive"} for col in set(df.columns) - {"ds", "y"}
        }
    )
    dates = make_dates_test()
    cleaning = make_cleaning_test()
    resampling = make_resampling_test()
    datasets = get_train_set(df, dates) if use_cv else get_train_val_sets(df, dates, config)
    datasets, models, forecasts = forecast_workflow(
        config, use_cv, make_future_forecast, cleaning, resampling, params, dates, datasets
    )
    # All dataframes in forecasts dictionary have at least a 'ds' column and a 'yhat' column
    assert all([len({"ds", "yhat"}.intersection(set(x.columns))) == 2 for x in forecasts.values()])
    # Models dictionary contains 2 models if a forecast has been made on future dates and 1 otherwise
    assert len(models) == 2 if make_future_forecast else 1
    # Training dataframe has at least 2 distinct dates
    assert datasets["train"].ds.nunique() > 1
    # Full dataframe has the same number of distinct dates as the input dataframe
    assert datasets["full"].ds.nunique() == df.ds.nunique()
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
        # Number of distinct dates in future dataframe is at least 1
        assert datasets["future"].ds.nunique() > 0
        # Number of distinct dates in future dataframe = number of distinct dates in future forecast dataframe
        assert forecasts["future"].ds.nunique() == datasets["future"].ds.nunique()
