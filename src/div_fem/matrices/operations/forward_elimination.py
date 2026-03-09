from div_fem.matrices import Matrix, _MatrixInputType, Vector, _VectorInputType


def forward_elimination(
    system_coeff: Matrix | _MatrixInputType, rhs_of_system: Vector | _VectorInputType
) -> Vector:
    if not isinstance(system_coeff, Matrix):
        system_coeff = Matrix(system_coeff)

    if not isinstance(rhs_of_system, Vector):
        rhs_of_system = Vector(rhs_of_system)

    if system_coeff.columns != rhs_of_system.rows:
        raise ValueError(
            f"An incorrect number of dimensions in the system was found. Matrix {system_coeff.shape}, rhs vector size {rhs_of_system.shape}"
        )

    partial_solution = Vector(rows=rhs_of_system.rows)

    partial_solution[0] = rhs_of_system[0]

    for i in range(1, rhs_of_system.rows):
        sum = 0
        for m in range(0, i):
            sum += system_coeff[i, m] * partial_solution[m]

        partial_solution[i] = rhs_of_system[i] - sum

    return partial_solution
