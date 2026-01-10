#!/usr/bin/env python3
"""
Module to concatenate two pandas DataFrames
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Task 11
    """
    df1 = index(df1)
    df2 = index(df2)

    df2_filtered = df2.loc[:1417411920]

    # Concatenate df2_filtered on top of df1
    # keys adds the top-level index labels
    df_combined = pd.concat([df2_filtered, df1], keys=['bitstamp', 'coinbase'])
    return df_combined
