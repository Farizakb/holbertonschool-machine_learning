#!/usr/bin/env python3
"""
Module to create a pandas DataFrame from a dictionary
"""
import pandas as pd


# Define the data for the dictionary
data = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
}

# Define the row labels
index_labels = ["A", "B", "C", "D"]

# Create the DataFrame and save it into the variable df
df = pd.DataFrame(data, index=index_labels)
