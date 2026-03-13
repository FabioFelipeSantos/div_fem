from div_fem.matrices.base_matrix import Matrix


class LocalStiffnessMatrix(Matrix):
    element_degree_of_freedom: int

    def __init__(self, element_degree_of_freedom: int) -> None:
        super().__init__(rows=element_degree_of_freedom)
