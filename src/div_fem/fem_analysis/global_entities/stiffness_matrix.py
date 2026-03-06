from __future__ import annotations
from typing import Self

from div_fem.matrices.base_matrix import Matrix


class GlobalStiffnessMatrix(Matrix):
    _initialized: bool = False
    _instance: GlobalStiffnessMatrix | None = None

    total_degree_of_freedom: int
    type_of_print_specifier = "scientific"

    def __new__(cls, total_degree_of_freedom: int) -> Self:
        if cls._instance:
            raise ValueError("The Global Stiffness matrix must be single")

        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, total_degree_of_freedom: int) -> None:
        if self._initialized:
            return

        self._initialized = True
        self.total_degree_of_freedom = total_degree_of_freedom

        super().__init__(rows=self.total_degree_of_freedom)
