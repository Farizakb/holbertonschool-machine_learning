#!/usr/bin/env python3
"""
Module to create a pandas DataFrame from a CSV file
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Creates a pd.DataFrame from a CSV file

    Args:
        filename: The path to the CSV file
        delimiter: The delimiter used in the CSV file
    Returns:
        The newly created pd.DataFrame
    """
    # Read the CSV file into a DataFrame using the specified delimiter
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
