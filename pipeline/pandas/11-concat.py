#!/usr/bin/env python3
"""
Module to concatenate two pandas DataFrames
"""
import pandas as pd
index = __import__('10-index').index


def contact(df1, df2):
    df1 = index(df1)
    df2 = index(df2)

    df = pd.concat([df2.loc[:1417411920], df1], keys=['bitstamp', 'coinbase'])
    return df
