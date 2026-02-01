#!/usr/bin/env python3
"""
Module to calculate the cofactor matrix of a matrix
"""
minor = __import__('1-minor').minor


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix:
    - matrix is a list of lists whose cofactor matrix should be calculated
    - If matrix is not a list of lists, raise a TypeError with the message
      matrix must be a list of lists
    - If matrix is not square or is empty, raise a ValueError with the message
      matrix must be a non-empty square matrix
    - Returns: the cofactor matrix of matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    minor_matrix = minor(matrix)
    cofactor_matrix = []
    for i in range(len(minor_matrix)):
        rows = [row for k, row in enumerate(minor_matrix) if k != i]
        cofactor_row = []
        for j in range(len(minor_matrix)):
            sub_matrix = [row[:j] + row[j+1:] for row in rows]
            cofactor_row.append(((-1) ** (i + j)) * determinant(sub_matrix))
        cofactor_matrix.append(cofactor_row)
    return cofactor_matrix
