from typing import List


def diff(data: List) -> List:
    """Return the element by element differences."""
    return [j-i for i, j in zip(data[:-1], data[1:])]
