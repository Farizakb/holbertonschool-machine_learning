#!/usr/bin/env python3
"""
Module to get the highest values in a pandas DataFrame
"""

def high(df):
    """ get the highest value of each column in a pandas dataframe """
    df = df.sort_values(by='High', ascending=False)
    return df
