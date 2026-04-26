from __future__ import annotations
from typing import Self

from div_fem.matrices.base_matrix import Matrix
from div_fem.matrices.main_types import _MatrixInputType
from div_fem.utils.descriptors.descriptor_private_name import DescriptorBaseClass


class GlobalStiffnessMatrix(Matrix):
    _initialized: bool = False
    _instance: GlobalStiffnessMatrix | None = None

    bandwidth = DescriptorBaseClass[int]()
    total_dof = DescriptorBaseClass[int]()
    type_of_print_specifier = "scientific"

    def __new__(cls, total_degree_of_freedom: int, bandwidth: int | None) -> Self:
        if cls._instance:
            raise ValueError("The Global Stiffness matrix must be single")

        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, total_degree_of_freedom: int, bandwidth: int | None) -> None:
        if self._initialized:
            return

        self._initialized = True

        self.total_dof = total_degree_of_freedom

        if bandwidth is not None:
            self.bandwidth = bandwidth
            super().__init__(rows=self.total_dof, columns=self.bandwidth)
        else:
            super().__init__(rows=self.total_dof)

    def assembly(
        self,
        idx: list[int] | tuple[list[int], list[int]],
        sub_matrix: Matrix | _MatrixInputType,
    ) -> None:
        if isinstance(idx, list):
            rows, columns = idx, idx
        else:
            rows, columns = idx

        if not isinstance(sub_matrix, Matrix):
            sub_matrix = Matrix(sub_matrix)

        for i, r in enumerate(rows):
            for j, c in enumerate(columns):
                self[r, c] += sub_matrix[i, j]

    def __getitem__(
        self,
        idx: tuple[int, int] | tuple[list[int], list[int]] | list[int],
    ) -> Matrix | float:
        if not isinstance(idx, tuple):
            rows = idx
            columns: list[int] = []

            for c, v in enumerate(rows):
                if rows[c] == v:
                    columns.append(0)
                else:
                    columns.append(v - rows[c])
