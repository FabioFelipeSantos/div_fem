from div_fem.matrices.base_matrix import Matrix


def matrix_multiplication(matrix_A: Matrix, matrix_B: Matrix) -> Matrix:
    m, l_A = matrix_A.shape
    l_B, n = matrix_B.shape

    if l_A != l_B:
        raise ValueError(
            f"To multiply two matrix the columns of the first must be the same of rows of the second. Received {l_A} as matrix A columns and {l_B} as matrix B rows"
        )

    l = l_A
    matrix_C = Matrix.zeros(m, n)

    for k in range(l):
        for i in range(m):
            x = matrix_A[i, k]
            if x != 0:
                for j in range(n):
                    y = matrix_B[k, j]
                    if y != 0:
                        matrix_C[i, j] += x * y

    return matrix_C
