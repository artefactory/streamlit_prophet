import pandas as pd
import pytest
from streamlit_prophet.lib.utils.load import download_toy_dataset, load_config

config, _ = load_config("config_streamlit.toml", "config_readme.toml")


@pytest.mark.parametrize(
    "url, date, target",
    [
        (
            config["datasets"][dataset]["url"],
            config["datasets"][dataset]["date"],
            config["datasets"][dataset]["target"],
        )
        for dataset in config["datasets"].keys()
    ],
)
def test_download_toy_dataset(url, date, target):
    output = download_toy_dataset(url)
    assert isinstance(output, pd.DataFrame)
    assert all([x in output.columns for x in [date, target]])
