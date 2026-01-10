#!/usr/bin/env python3
"""
Module to set the index of a pandas DataFrame
"""


def index(df):
    """ set the index of a pandas dataframe to the Date column """
    df = df.set_index('Timestamp')
    return df
