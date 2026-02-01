#!/usr/bin/env python3

determinant = __import__('0-determinant').determinant

if __name__ == "__main__":
    # Test 0x0
    print(f"det([[]]) = {determinant([[]])}") # Expected: 1
    
    # Test 1x1
    print(f"det([[5]]) = {determinant([[5]])}") # Expected: 5
    
    # Test 2x2
    m2 = [[1, 2], [3, 4]]
    print(f"det({m2}) = {determinant(m2)}") # Expected: -2
    
    # Test 3x3
    m3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(f"det({m3}) = {determinant(m3)}") # Expected: 0
    
    # Test 3x3 invertible
    m3_inv = [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
    print(f"det({m3_inv}) = {determinant(m3_inv)}") # Expected: -306 (manual calc check)
    # 6*(-14-40) - 1*(28-10) + 1*(32+4) = 6*(-54) - 18 + 36 = -324 - 18 + 36 = -306. Correct.

    # Test TypeErrors
    try:
        determinant("not a list")
    except TypeError as e:
        print(f"TypeError caught: {e}")
        
    try:
        determinant([1, 2, 3])
    except TypeError as e:
        print(f"TypeError caught: {e}")

    # Test ValueErrors
    try:
        determinant([[1, 2], [3]])
    except ValueError as e:
        print(f"ValueError caught: {e}")
