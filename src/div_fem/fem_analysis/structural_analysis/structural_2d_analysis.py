from __future__ import annotations
from typing import Any


from div_fem.fem_analysis.structural_analysis.structural_analysis_interface import (
    StructuralAnalysisInterface,
)


class Structural2DAnalysis(StructuralAnalysisInterface):
    max_bandwidth: int
    total_degree_of_freedom: int
    global_stiffness_matrix: None

    def __call__(self) -> Any:
        self.total_degree_of_freedom = self._total_dof_calculation()
        self.renumber_dofs()
        self.max_bandwidth = self._bandwidth_calculation()

    def renumber_dofs(self) -> None:
        new_order = self.elements.reverse_cuthill_mckee()

        for new_idx, old_node_index in enumerate(new_order):
            point = self.points(old_node_index)
            point.dof_numbers = self.points.calculating_dof_numbers(new_idx)

    def _bandwidth_calculation(self) -> int:
        all_elements_bandwidths: list[int] = []

        for element in self.elements:
            element_dof = element.degree_of_freedom
            all_elements_bandwidths.append(max(element_dof) - min(element_dof))

        return max(all_elements_bandwidths)

    def _total_dof_calculation(self) -> int:
        number_of_points = self.points.number_of_points
        number_of_dof_per_node = self.points.dof_per_node

        return number_of_points * number_of_dof_per_node
