#!/usr/bin/env python3
"""
Module to calculate the adjugate matrix of a matrix
"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix:
    - matrix is a list of lists whose adjugate matrix should be calculated
    - If matrix is not a list of lists, raise a TypeError with the message
      matrix must be a list of lists
    - If matrix is not square or is empty, raise a ValueError with the message
      matrix must be a non-empty square matrix
    - Returns: the adjugate matrix of matrix
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

    # The adjugate is the transpose of the cofactor matrix
    cofactor_matrix = cofactor(matrix)

    # Transpose the cofactor matrix to get the adjugate
    adjugate_matrix = [
        [cofactor_matrix[i][j] for i in range(n)] for j in range(n)
    ]

    return adjugate_matrix