#!/usr/bin/env python3
"""
Module to rename a column and convert its values to datetime
"""
import pandas as pd


def rename(df):
    """
    Renames the column Timestamp to Datetime, converts values to datetime,
    and returns only Datetime and Close columns.

    Args:
        df: pd.DataFrame containing a Timestamp column

    Returns:
        The modified pd.DataFrame
    """
    # Rename the Timestamp column to Datetime
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convert the Datetime column from Unix timestamp to datetime objects
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Return only the Datetime and Close columns
    return df[['Datetime', 'Close']]
