#!/usr/bin/env python3
"""
Module to flip the values of a binary switch column in a pandas DataFrame
"""


def flip_switch(df):
    """ sort and tranpose a pandas dataframe """
    df = df.sort_index(ascending=False).T
    return df
