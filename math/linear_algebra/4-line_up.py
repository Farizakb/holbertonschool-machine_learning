#!/usr/bin/env python3
""" A script that adds two arrays element-wise"""

def add_arrays(arr1, arr2):
    new_list = []
    for i in range(len(arr1)):
        new_list.append(arr1[i] + arr2[i])
    return new_list
