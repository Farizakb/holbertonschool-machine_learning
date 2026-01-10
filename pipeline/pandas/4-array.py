#!/usr/bin/env python3
"""
Module to convert specific parts of a DataFrame to a numpy array
"""
import pandas as pd


def array(df):
    """
    Takes a pd.DataFrame and returns the last 10 rows
    of the High and Close columns as a numpy.ndarray

    Args:
        df: pd.DataFrame containing columns High and Close

    Returns:
        numpy.ndarray containing the selected values
    """
    selected_data = df[['High', 'Close']].tail(10)
    return selected_data.to_numpy()
