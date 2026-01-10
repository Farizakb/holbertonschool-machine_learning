#!/usr/bin/env python3
"""
Module to convert specific parts of a DataFrame to a numpy array
"""


def array(df):
    """
    Takes a pd.DataFrame and returns the last 10 rows
    of the High and Close columns as a numpy.ndarray

    Args:
        df: pd.DataFrame containing columns High and Close

    Returns:
        numpy.ndarray containing the selected values
    """
    # Select the High and Close columns, then take the last 10 rows
    # .tail(10) is a convenient way to get the last N rows
    selected_data = df[['High', 'Close']].tail(10)

    # Convert the resulting selection into a numpy array
    return selected_data.to_numpy()