from div_fem.matrices import Matrix, Vector, _MatrixInputType, _VectorInputType

from .lu_decomposition import LU_decomposition
from .back_substitution import back_substitution
from .forward_elimination import forward_elimination


def solve_lin_system(
    coeff_matrix: Matrix | _MatrixInputType, rhs_system: Vector | _VectorInputType
) -> Vector:
    L, U = LU_decomposition(coeff_matrix)
    partial_solution = forward_elimination(L, rhs_system)
    return back_substitution(U, partial_solution)
