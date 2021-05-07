import pytest
from streamlit_prophet.lib.utils.misc import reverse_list


@pytest.mark.parametrize(
    "L, N, expected",
    [
        ([1, 2, 3, 4, 5], 3, [3, 2, 1]),
        ([1, 2, 3, 4, 5], 10, [5, 4, 3, 2, 1]),
        ([1, 2, 3, 4, 5], 5, [5, 4, 3, 2, 1]),
    ],
)
def test_reverse_list(L, N, expected):
    # The output list has the expected elements in the right order
    assert reverse_list(L, N) == expected
