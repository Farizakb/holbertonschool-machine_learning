#!/usr/bin/env python3
"""Module for calculating the derivative of a polynomial."""

def poly_derivative(poly):
    """Calculates the derivative of a polynomial.

    Args:
        poly (list): A list of coefficients representing the polynomial,
                     where the index represents the power of x.

    Returns:
        list: A list of coefficients representing the derivative of
        the polynomial.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    derivative = []
    for power, coeff in enumerate(poly):
        if power > 0:
            derivative.append(coeff * power)

    return derivative if derivative else [0]
