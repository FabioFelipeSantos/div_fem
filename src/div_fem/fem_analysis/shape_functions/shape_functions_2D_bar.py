from div_fem.matrices.base_matrix import Matrix


def shape_functions_2D_bar(zeta: float) -> Matrix:
    N1 = 1 - zeta
    N2 = zeta

    return Matrix([[N1, N2]])


def first_derivative_shape_functions_2D_bar(zeta: float) -> Matrix:
    dN1 = -1
    dN2 = 1

    return Matrix([[dN1, dN2]])
