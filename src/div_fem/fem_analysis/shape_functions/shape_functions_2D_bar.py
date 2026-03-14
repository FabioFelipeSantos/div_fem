from div_fem.matrices.base_matrix import Matrix


def shape_functions_2D_bar(xi: float) -> Matrix:
    N1 = (1 - xi) / 2
    N2 = (1 + xi) / 2

    return Matrix([[N1, N2]])


def first_derivative_shape_functions_2D_bar(xi: float) -> Matrix:
    dN1 = -1 / 2
    dN2 = 1 / 2

    return Matrix([[dN1, dN2]])
