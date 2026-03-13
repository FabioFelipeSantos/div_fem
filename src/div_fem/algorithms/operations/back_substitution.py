from div_fem.matrices.base_matrix import Matrix, _MatrixInputType
from div_fem.matrices.base_vector import Vector, _VectorInputType


def back_substitution(
    U_part: Matrix | _MatrixInputType, partial_solution: Vector | _VectorInputType
) -> Vector:
    if not isinstance(U_part, Matrix):
        U_part = Matrix(U_part)

    if not isinstance(partial_solution, Vector):
        partial_solution = Vector(partial_solution)

    if U_part.columns != partial_solution.rows:
        raise ValueError(
            f"An incorrect number of dimensions in the system was found. Matrix {U_part.shape}, rhs vector size {partial_solution.shape}"
        )

    final_solution = Vector(rows=partial_solution.rows)

    last_index = final_solution.rows - 1
    final_solution[last_index] = (
        partial_solution[last_index] / U_part[last_index, last_index]
    )

    for i in range(last_index - 1, -1, -1):
        sum = 0
        for m in range(i + 1, last_index + 1):
            sum += U_part[i, m] * final_solution[m]

        final_solution[i] = (1 / U_part[i, i]) * (partial_solution[i] - sum)

    return final_solution
