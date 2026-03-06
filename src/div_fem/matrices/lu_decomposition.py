def lu_decomposition(
    matrix: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    rows = len(matrix)
    columns = len(matrix[0])

    if rows != columns:
        raise ValueError(
            f"To construct a LU triangularization the matrix must be square. Number of rows: {rows}; Number of columns: {columns}"
        )

    for j in range(1, columns):
        matrix[j][0] = matrix[j][0] / matrix[0][0]

        if j > 1:
            for i in range(1, j):
                summation1 = 0
                summation2 = 0
                for m in range(0, i):
                    summation1 += matrix[j][m] * matrix[m][i]
                    summation2 += matrix[i][m] * matrix[m][j]

                    matrix[j][i] = (1 / matrix[i][i]) * (matrix[j][i] - summation1)
                    matrix[i][j] = matrix[i][j] - summation2

                summation = 0
                for m in range(0, j):
                    summation += matrix[j][m] * matrix[m][j]
                matrix[j][j] = matrix[j][j] - summation

    return (_get_l(matrix, rows, columns), _get_u(matrix, rows, columns))


def _get_l(matrix: list[list[float]], rows: int, columns: int) -> list[list[float]]:
    l: list[list[float]] = []
    for i in range(0, rows):
        row_l = []
        for j in range(0, columns):
            if i == j:
                row_l.append(1)
            elif j > i:
                row_l.append(0)
            else:
                row_l.append(matrix[i][j])
        l.append(row_l)
    return l


def _get_u(matrix: list[list[float]], rows: int, columns: int) -> list[list[float]]:
    u: list[list[float]] = []
    for j in range(0, columns):
        row_u = []
        for i in range(0, rows):
            if j > i:
                row_u.append(0)
            else:
                row_u.append(matrix[i][j])
        u.append(row_u)
    return u
