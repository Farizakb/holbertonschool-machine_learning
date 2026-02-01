#!/usr/bin/env python3
"""
Module to calculate the minor matrix of a matrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix:
    - matrix is a list of lists whose determinant should be calculated
    - If matrix is not a list of lists, raise a TypeError with the message
      matrix must be a list of lists
    - If matrix is not square, raise a ValueError with the message
      matrix must be a square matrix
    - The list [[]] represents a 0x0 matrix
    - Returns: the determinant of matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        if matrix == []:
            return 1
        raise TypeError("matrix must be a list of lists")

    # Special case for 0x0 matrix represented as [[]]
    if matrix == [[]]:
        return 1

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    # Base case for 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    def get_minor(m, j):
        """
        Helper to get the minor of a matrix after removing first row and col j
        """
        return [row[:j] + row[j+1:] for row in m[1:]]

    det = 0
    for j in range(n):
        det += ((-1) ** j) * matrix[0][j] * determinant(get_minor(matrix, j))

    return det


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
