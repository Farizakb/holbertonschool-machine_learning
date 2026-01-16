#!/usr/bin/env python3

"""Module for calculating the summation of i squared from 1 to n."""


def summation_i_squared(n):
    """Calculates the summation of i squared from 1 to n.

    Args:
        n (int): The upper limit of the summation.
    Returns:
        int: The summation of i squared from 1 to n.
    """
    if type(n) is not int or n < 1:
        return None
    result = (n * (n + 1) * (2 * n + 1)) / 6
    return int(result)
