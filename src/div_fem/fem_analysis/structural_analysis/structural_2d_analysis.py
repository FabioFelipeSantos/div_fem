from __future__ import annotations
from typing import TYPE_CHECKING, Any, Self


from div_fem.fem_analysis.structural_analysis.structural_analysis_interface import StructuralAnalysisInterface


class Structural2DAnalysis(StructuralAnalysisInterface):

    def __call__(self) -> Any:
        dof = self.dof
        points = self.points
        elements = self.elements
        boundaries_cond = self.boundaries_cond

        print(dof)
        print(points)
        print(elements)
        print(boundaries_cond)
