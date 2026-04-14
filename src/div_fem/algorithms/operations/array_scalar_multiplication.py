from div_fem.matrices.base_matrix import Matrix
from div_fem.matrices.base_vector import Vector


def array_scalar_multiplication(array: Matrix | Vector, scalar: float) -> Matrix | Vector:

    if isinstance(array, Matrix):
        return _matrix_scalar_multiplication(array, scalar)
    else:
        return _vector_scalar_multiplication(array, scalar)


def _matrix_scalar_multiplication(matrix: Matrix, scalar: float) -> Matrix:
    m, n = matrix.shape

    result = Matrix.zeros(m, n)

    if not scalar:
        return result
    elif abs(scalar - 1.0) < 1e-12:
        return matrix
    else:
        for i in range(m):
            for j in range(n):
                result[i, j] = matrix[i, j] * scalar
        return result


def _vector_scalar_multiplication(vector: Vector, scalar: float) -> Vector:
    m = vector.rows

    result = Vector.zeros(m)

    if not scalar:
        return result
    elif abs(scalar - 1.0) < 1e-12:
        return vector
    else:
        for i in range(m):
            result[i] = vector[i] * scalar
        return result
