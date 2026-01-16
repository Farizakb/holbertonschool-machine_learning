#!/usr/bin/env python3
"""Module for calculating the integral of a polynomial."""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial.

    Args:
        poly (list): A list of coefficients representing the polynomial,
                     where the index represents the power of x.
        C (int or float): The constant of integration.

    Returns:
        list: A list of coefficients representing the integral of
        the polynomial.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    integral = [C]
    for power, coeff in enumerate(poly):
        integral.append(coeff / (power + 1))

    return integral
