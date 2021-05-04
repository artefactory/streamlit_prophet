from datetime import datetime

import pytest
from streamlit_prophet.render.dialogs import render_clock, render_hello


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Jeanette", "Hello Jeanette!"),
        ("Raven", "Hello Raven!"),
        ("Maxine", "Hello Maxine!"),
        ("Matteo", "Hello Matteo!"),
        ("Destinee", "Hello Destinee!"),
        ("Alden", "Hello Alden!"),
        ("Mariah", "Hello Mariah!"),
        ("Anika", "Hello Anika!"),
        ("Isabella", "Hello Isabella!"),
    ],
)
def test_render_hello(name, expected):
    """Example test with parametrization."""
    assert render_hello(name) == expected


def test_render_clock():
    """Example test with error check."""
    time = render_clock()
    assert time.startswith("It is ")
    assert time.endswith("!")
    try:
        assert isinstance(datetime.strptime(time[6:-1], "%c"), datetime)
    except ValueError as err:
        pytest.fail(str(err))
