import numpy as np
import torch

def rowswap(matrix: torch.Tensor, src_row: int, target_row: int) -> torch.Tensor:
    matrix_copy = matrix.clone()
    matrix_copy[[src_row, target_row]] = matrix_copy[[target_row, src_row]]
    return matrix_copy

def rowscale(matrix: torch.Tensor, src_row: int, scale_factor: float) -> torch.Tensor:
    matrix_copy = matrix.clone().float()
    matrix_copy[src_row] = matrix_copy[src_row] * scale_factor
    return matrix_copy

def rowreplacement(matrix: torch.Tensor, first_row: int, second_row: int, j: int, k: int) -> torch.Tensor:
    matrix_copy = matrix.clone().float()
    matrix_copy = rowscale(matrix_copy, first_row, j)
    matrix_copy[first_row] = matrix_copy[first_row] + (matrix_copy[second_row] * k)
    return matrix_copy

def rref(matrix: torch.Tensor) -> torch.Tensor:
    # Copy matrix and enable the use of float numbers
    matrix_copy = matrix.clone().float()

    # Get the number of rows and columns 
    rows, cols = matrix_copy.shape

    # Initialize pivot row
    pivot_row = 0

    # For each column
    for col in range(cols):

        # Break once all rows have been evaluated
        if pivot_row >= rows:
            break


        max_row = pivot_row

        # Check every row below pivot row
        for r in range(pivot_row + 1, rows):

            # If value in a row is bigger than max row, set new max row
            if abs(matrix_copy[r,col]) > abs(matrix_copy[max_row, col]):
                max_row = r

        # If all rows below and including pivot row have value of 0, move to next column
        if abs(matrix_copy[max_row, col]) < 1e-7:
            continue

        # Set max_row as the new pivot
        if max_row != pivot_row:
            matrix_copy = rowswap(matrix_copy, pivot_row, max_row)

        # Extract the value in the pivot row
        pivot_val = matrix_copy[pivot_row, col].item()

        # Use reciprocal to make pivot row value = 1
        matrix_copy = rowscale(matrix_copy, pivot_row, 1.0/pivot_val)

        # Scan all the rows
        for r in range(rows):

            # If row isn't pivot, find value that will make row 0 when subtracted
            if r != pivot_row:
                factor = -matrix_copy[r, col].item()

                # If row/factor isn't already 0, add the 1 from pivot row times the factor to the selected row to make it zero
                if abs(factor) > 1e-7:
                    matrix_copy = rowreplacement(matrix_copy, r, pivot_row, 1.0, factor)

        # Increment pivot row
        pivot_row += 1

    # Clear up any residual near-zero values and return RREF matrix
    matrix_copy[torch.abs(matrix_copy) < 1e-7] = 0.0
    return matrix_copy

if __name__ == "__main__":
    # Test Matrix from Step 8
    M = torch.tensor([
        [1.0, 3.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 1.0, 0.0, 9.0],
        [0.0, 0.0, 0.0, 1.0, -4.0]
    ])

    print("Initial Matrix:")
    print(M)
    print("-" * 40)

    # 1. Perform elementary operation R1 <-> R2 using rowswap
    # Note: R1 corresponds to index 0, R2 corresponds to index 1
    step1_mat = rowswap(M, 0, 1)
    print("Step 1: R1 <-> R2")
    print(step1_mat)
    print("-" * 40)

    # 2. Perform elementary operation (1/3)R1 using rowscale on the resulting matrix
    step2_mat = rowscale(step1_mat, 0, 1.0 / 3.0)
    print("Step 2: (1/3) * R1")
    print(step2_mat)
    print("-" * 40)

    # 3. Perform elementary operation R3 = -3*R1 + R3 using rowreplacement on the resulting matrix
    # Formula: R_i = j*R_i + k*R_j -> R3 = 1.0*R3 + (-3.0)*R1
    # Note: R3 is index 2, R1 is index 0
    step3_mat = rowreplacement(step2_mat, 2, 0, 1.0, -3.0)
    print("Step 3: R3 = -3*R1 + R3")
    print(step3_mat)
    print("-" * 40)
