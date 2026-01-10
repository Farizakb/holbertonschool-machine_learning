#!/usr/bin/env python3
"""
Module to prune rows with NaN values in a pandas DataFrame
"""


def prune(df):
    """ remove all rows with NaN in the Close column """
    df = df.dropna(subset=['Close'])
    return df
