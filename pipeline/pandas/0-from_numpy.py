#!/usr/bin/env python3
"""
Module for converting a numpy array to a pandas DataFrame
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray

    Args:
        array: np.ndarray from which to create the pd.DataFrame

    Returns:
        The newly created pd.DataFrame with alphabetized column labels
    """
    # Get the number of columns in the array
    num_cols = array.shape[1]

    # Generate column labels: A, B, C... up to the number of columns
    # chr(65) is 'A', chr(66) is 'B', etc.
    col_names = [chr(65 + i) for i in range(num_cols)]

    # Create and return the DataFrame
    return pd.DataFrame(array, columns=col_names)
