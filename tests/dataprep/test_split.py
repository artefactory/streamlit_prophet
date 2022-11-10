from datetime import timedelta

import pytest
from streamlit_prophet.lib.dataprep.split import get_cv_cutoffs
from streamlit_prophet.lib.utils.load import load_config
from tests.samples.dict import make_dates_test

config, _, _ = load_config(
    "config_streamlit.toml", "config_instructions.toml", "config_readme.toml"
)


# Temporarily deactivate this test as script_runner is deprecated
# @pytest.mark.parametrize(
#   "train_start, train_end, val_start, val_end, freq",
#   [
#       ("2015-01-01", "2019-12-31", "2020-01-01", "2023-01-01", "Y"),
#       ("2015-01-01", "2019-12-31", "2020-01-01", "2020-01-15", "M"),
#       ("2020-01-01", "2020-01-15", "2021-01-01", "2021-01-15", "D"),
#       ("2020-01-01", "2020-12-31", "2021-01-01", "2021-01-01", "D"),
#       ("2020-01-01", "2020-12-31", "2021-01-01", "2021-01-05", "W"),
#       ("2020-01-01 00:00:00", "2020-01-01 12:00:00", "2021-01-01", "2021-01-05", "H"),
#   ],
# )
# def test_raise_error_train_val_dates(train_start, train_end, val_start, val_end, freq):
#    train = pd.date_range(start=train_start, end=train_end, freq=freq)
#    val = pd.date_range(start=val_start, end=val_end, freq=freq)
#    # Streamlit should stop and display an error message
#    with pytest.raises(st.script_runner.StopException):
#        raise_error_train_val_dates(val, train, config=config, dates=make_dates_test())

# Temporarily deactivate this test as script_runner is deprecated
# @pytest.mark.parametrize(
#    "dates",
#    [
#        (
#            make_dates_test(
#                train_start="2020-01-01",
#                train_end="2021-01-01",
#                n_folds=12,
#                folds_horizon=30,
#                freq="D",
#            )
#        ),
#        (
#            make_dates_test(
#                train_start="2020-01-01",
#                train_end="2021-01-01",
#                n_folds=5,
#                folds_horizon=3,
#                freq="4D",
#            )
#        ),
#        (
#            make_dates_test(
#                train_start="2020-01-01",
#                train_end="2021-01-01",
#                n_folds=50,
#                folds_horizon=1,
#                freq="W",
#            )
#        ),
#        (
#            make_dates_test(
#                train_start="2020-01-01",
#                train_end="2020-01-02",
#                n_folds=7,
#                folds_horizon=3,
#                freq="H",
#            )
#        ),
#    ],
# )
# def test_raise_error_cv_dates(dates):
#    # Streamlit should stop and display an error message
#    with pytest.raises(st.script_runner.StopException):
#        raise_error_cv_dates(dates, resampling={"freq": dates["freq"]}, config=config)


@pytest.mark.parametrize(
    "dates",
    [
        (
            make_dates_test(
                train_start="1900-01-01",
                train_end="2021-01-01",
                n_folds=5,
                folds_horizon=3,
                freq="Y",
            )
        ),
        (
            make_dates_test(
                train_start="1950-01-01",
                train_end="2021-01-01",
                n_folds=5,
                folds_horizon=4,
                freq="Q",
            )
        ),
        (
            make_dates_test(
                train_start="1970-01-01",
                train_end="2021-01-01",
                n_folds=5,
                folds_horizon=6,
                freq="M",
            )
        ),
        (make_dates_test(freq="W")),
        (make_dates_test(freq="D")),
        (
            make_dates_test(
                train_start="2021-01-01",
                train_end="2021-01-30",
                n_folds=5,
                folds_horizon=12,
                freq="H",
            )
        ),
    ],
)
def test_get_cv_cutoffs(dates):
    output = sorted(get_cv_cutoffs(dates, freq=dates["freq"][-1]))
    # Output list should have the number of elements specified by the 'n_folds' value of dates dictionary
    assert len(output) == dates["n_folds"]
    # Maximum cutoff date + Horizon should be a date before training end date
    assert max(output) + timedelta(days=(output[-1] - output[-2]).days) <= dates["train_end_date"]
