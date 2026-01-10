#!/usr/bin/env python3
"""
Module to slice specific data from a pandas DataFrame
"""


def slice(df):
    """
    Takes a pd.DataFrame and returns the row
    with index label 60 and the columns High, Low, Close, and Volume_BTC
    Args:
        df: pd.DataFrame containing the relevant columns
    Returns:
        pd.DataFrame sliced to the specified row and columns
    """
    df = df.loc[::60, ['High', 'Low', 'Close', 'Volume_(BTC)']]
    return df
