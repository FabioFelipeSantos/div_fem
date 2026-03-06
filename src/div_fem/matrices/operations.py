def multiplication(
    matrixA: list[list[float]], matrixB: list[list[float]]
) -> list[list[float]]:
    c: list[list[float]] = [[]]

    for i in range(0, len(matrixA)):
        row_c = []
        for j in range(0, len(matrixB[0])):
            c_ij = 0
            for k in range(0, len(matrixB)):
                c_ij += matrixA[i][k] * matrixB[k][j]
            row_c.append(c_ij)
        c.append(row_c)
    return c
