from typing import List


def diff(data: List) -> List:
    """Return the element by element differences.

    E.g.
    diff([1, 2, 3, 5, 8, 13]) -> [1, 1, 2, 3, 5]
    """
    return [j-i for i, j in zip(data[:-1], data[1:])]
