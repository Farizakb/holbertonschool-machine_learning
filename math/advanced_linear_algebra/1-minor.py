#!/usr/bin/env python3
"""
Module to calculate the minor matrix of a matrix
"""
from 0-determinant import determinant


def minor(matrix):
    """
    Calculates the minor matrix of a matrix:
    - matrix is a list of lists whose minor matrix should be calculated
    - If matrix is not a list of lists, raise a TypeError with the message
      matrix must be a list of lists
    - If matrix is not square or is empty, raise a ValueError with the message
      matrix must be a non-empty square matrix
    - Returns: the minor matrix of matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minor_matrix = []
    for i in range(n):
        rows = [row for k, row in enumerate(matrix) if k != i]
        minor_row = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:] for row in rows]
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix
