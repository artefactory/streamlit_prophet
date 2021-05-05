def reverse_list(L: list, N: int) -> list:
    if N < len(L):
        L = L[:N]
    reversed_list = [L[len(L) - 1 - i] for i, x in enumerate(L)]
    return reversed_list
