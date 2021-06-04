from typing import Any, List


def reverse_list(L: List[Any], N: int) -> List[Any]:
    """Cuts the list after the N-th element and reverses its order.

    Parameters
    ----------
    L : list
        List to be reversed.
    N : int
        Index at which the list will be cut if it is smaller than the list length.

    Returns
    -------
    list
        Reversed list.
    """
    if N < len(L):
        L = L[:N]
    reversed_list = [L[len(L) - 1 - i] for i, x in enumerate(L)]
    return reversed_list
