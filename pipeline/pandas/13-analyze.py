#!/usr/bin/env python3
"""
Module to analyze a pandas DataFrame
"""


def analyze(df):
    """ analyze the dataframe """
    df = df.drop(columns=['Timestamp']).describe()
    return df
