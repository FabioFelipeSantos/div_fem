from __future__ import annotations
from typing import TYPE_CHECKING, Any, Self
from abc import ABC, abstractmethod

from div_fem.utils.descriptors.descriptor_private_name import (
    PrivateName,
    DescriptorBaseClass,
)

if TYPE_CHECKING:
    from div_fem.fem_analysis.geometry.points import Points
    from div_fem.fem_analysis.geometry.elements_container import Elements
    from div_fem.fem_analysis.geometry.boundaries_conditions import BoundariesConditions


class StructuralPoints(PrivateName):

    def __get__(self, obj: StructuralAnalysisInterface, objtype=None) -> Points:
        return getattr(obj, self.private_name)

    def __set__(self, obj: StructuralAnalysisInterface, value: Points) -> None:
        if value.number_of_points < 2:
            raise ValueError(
                "Expected that number of points be greater than zero since any 2D element must have a starting and ending points"
            )

        setattr(obj, self.private_name, value)
        value.structural_analysis = obj


class StructuralElements(PrivateName):

    def __get__(self, obj: StructuralAnalysisInterface, objtype=None) -> Elements:
        return getattr(obj, self.private_name)

    def __set__(self, obj: StructuralAnalysisInterface, value: Elements) -> None:
        if value.number_of_elements < 1:
            raise ValueError(
                f"Expected one or more elements. Received {value.number_of_elements}"
            )

        setattr(obj, self.private_name, value)
        value.structural_analysis = obj


class StructuralBoundaryConditions(PrivateName):

    def __get__(
        self, obj: StructuralAnalysisInterface, objtype=None
    ) -> BoundariesConditions:
        return getattr(obj, self.private_name)

    def __set__(
        self, obj: StructuralAnalysisInterface, value: BoundariesConditions
    ) -> None:
        if value.count < 1:
            raise ValueError(
                f"Expected one or more boundaries conditions. Received {value.count}"
            )

        setattr(obj, self.private_name, value)


class StructuralAnalysisInterface:
    _initialized: bool = False
    _instance: StructuralAnalysisInterface | None = None

    points = StructuralPoints()
    elements = StructuralElements()
    boundaries_cond = StructuralBoundaryConditions()

    def __new__(cls) -> Self:
        if cls._instance:
            raise ValueError(
                "The StructuralAnalysisInterface class is single to the analysis."
            )

        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True

    @property
    def number_points(self) -> int:
        return self.points.number_of_points

    @property
    def number_elements(self) -> int:
        return self.elements.number_of_elements

    @abstractmethod
    def __call__(self) -> Any:
        pass
