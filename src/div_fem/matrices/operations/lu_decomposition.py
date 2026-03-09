from div_fem.matrices import Matrix, _MatrixInputType


def LU_decomposition(matrix: Matrix | _MatrixInputType) -> tuple[Matrix, Matrix]:
    if not isinstance(matrix, Matrix):
        matrix = Matrix(matrix)
    if matrix.rows != matrix.columns:
        raise ValueError(
            f"A LU decomposition is only available for square matrices. Received a {matrix.rows} x {matrix.columns} matrix."
        )

    L = Matrix(rows=matrix.rows)
    U = Matrix(rows=matrix.rows)

    L[0, 0] = 1
    U[0, 0] = matrix[0, 0]

    for j in range(1, matrix.rows):
        L[j, 0] = matrix[j, 0] / U[0, 0]
        U[0, j] = matrix[0, j]

        if j > 1:
            for i in range(1, j):
                sum1 = 0
                sum2 = 0

                for m in range(0, i):
                    sum1 += L[j, m] * U[m, i]
                    sum2 += L[i, m] * U[m, j]

                L[j, i] = (1 / U[i, i]) * (matrix[j, i] - sum1)
                U[i, j] = matrix[i, j] - sum2

        sum = 0
        for m in range(j):
            sum += L[j, m] * U[m, j]

        U[j, j] = matrix[j, j] - sum
        L[j, j] = 1

    return (L, U)
