from div_fem.matrices.base_vector import Vector


class LocalForcesVector(Vector):
    element_degree_of_freedom: int

    def __init__(self, element_degree_of_freedom: int) -> None:
        super().__init__(rows=element_degree_of_freedom)
