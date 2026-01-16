#!/usr/bin/env python3
"""Module for calculating the integral of a polynomial."""


def poly_integral(poly, C=0):
    # Check if poly is a valid non-empty list
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    
    # Check if C is an integer
    if not isinstance(C, int):
        return None
        
    # Check if all elements in poly are numbers
    if not all(isinstance(x, (int, float)) for x in poly):
        return None

    # Start the new polynomial with the constant of integration at index 0
    # The integral of ax^n is (a / n+1) * x^(n+1)
    integral = [C]
    
    for i in range(len(poly)):
        new_coeff = poly[i] / (i + 1)
        
        # Convert to integer if it's a whole number (e.g., 4.0 -> 4)
        if new_coeff == int(new_coeff):
            new_coeff = int(new_coeff)
            
        integral.append(new_coeff)

    # Remove trailing zeros to keep the list as small as possible, 
    # but ensure the list has at least one element.
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
        
    return integral
