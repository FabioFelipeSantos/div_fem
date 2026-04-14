from div_fem.matrices.base_matrix import Matrix
from div_fem.matrices.base_vector import Vector


def matrix_vector_multiplication(matrix: Matrix, vector: Vector) -> Vector:
    m, n_m = matrix.shape
    n_v = vector.rows

    if n_m != n_v:
        raise ValueError(
            f"To multiply a matrix and a vector the columns of the matrix must be the same of rows of the vector. Received {n_m} as matrix columns and {n_v} as vector rows"
        )

    result = Vector.zeros(m)

    for k in range(n_v):
        y = vector[k]
        if y != 0:
            for i in range(m):
                x = matrix[i, k]
                if x != 0:
                    result[i] += x * y

    return result
