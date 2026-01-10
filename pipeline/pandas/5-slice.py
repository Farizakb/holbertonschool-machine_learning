#!/usr/bin/env python3
"""
Module to slice a pandas DataFrame
"""

def slice(df):
    """
    Slices a pd.DataFrame to specific columns and every 60th row

    Args:
        df: the pd.DataFrame to slice

    Returns:
        The sliced pd.DataFrame
    """
    # Select the required columns first
    # Then use positional slicing [::60] to get every 60th row
    df_sliced = df[['High', 'Low', 'Close', 'Volume_BTC']].iloc[::60]

    return df_sliced