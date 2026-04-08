from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Self, Union, overload

from div_fem.fem_analysis.geometry.elements.element_load_interface import (
    ElementLoadInterface,
)
from div_fem.fem_analysis.geometry.elements_interface import ElementInterface
from div_fem.fem_analysis.structural_analysis.structural_analysis_interface import (
    StructuralAnalysisInterface,
)
from div_fem.utils.descriptors.descriptor_private_name import DescriptorBaseClass

if TYPE_CHECKING:
    from div_fem.fem_analysis.geometry.point import Point

_ContainerElement = ElementInterface[
    str,
    Mapping[str, Any],
    Mapping[str, Any],
    ElementLoadInterface[
        str,
        Any,
        Union[Mapping[str, Any], float],
        Union[float, "Point", None],
    ],
]


class StructuralAnalysisDescriptor(DescriptorBaseClass[StructuralAnalysisInterface]):

    def __set__(self, obj: "Elements", value: StructuralAnalysisInterface) -> None:
        super().__set__(obj, value)

        try:
            points_class = value.points

            for element in obj:
                if not element.interpolation_points_already_put_in_points_class:
                    points_class(element.interpolation_points)
                    element.interpolation_points_already_put_in_points_class = True
        except:
            raise


class Elements:
    _initialized: bool = False
    _instance: Elements | None = None

    structural_analysis = StructuralAnalysisDescriptor()

    _elements: list[_ContainerElement] = []

    _index_next_element: int = 1

    _number_of_interpolation_points: int

    def __new__(
        cls,
    ) -> Self:
        if cls._instance:
            raise ValueError("The Elements object must be single per analysis.")

        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
    ) -> None:
        if self._initialized:
            return

        self._initialized = True

    @property
    def number_of_elements(self) -> int:
        return self._index_next_element - 1

    @overload
    def __call__(self, elements_or_index: int) -> _ContainerElement: ...

    @overload
    def __call__(
        self, elements_or_index: _ContainerElement | list[_ContainerElement]
    ) -> None: ...

    def __call__(
        self, elements_or_index: int | _ContainerElement | list[_ContainerElement]
    ) -> _ContainerElement | None:
        if isinstance(elements_or_index, int):
            return self._elements[elements_or_index - 1]
        else:
            self.add(elements_or_index)

    def add(self, elements: _ContainerElement | list[_ContainerElement]) -> None:
        if isinstance(elements, list):
            for element in elements:
                self._adding_one_element(element)
        else:
            self._adding_one_element(elements)

        return

    def _adding_one_element(self, element: _ContainerElement) -> None:
        element.elements_container = self
        element.index = self._index_next_element
        self._elements.append(element)
        self._index_next_element += 1

    def print(self) -> None:
        print(str(self))
        return

    def _string_for_element_info(self, element: _ContainerElement) -> str:
        begin = f"Element[{element.index:>3d}]: (\n    "
        element_data: list[str] = []

        extreme_points = element.extreme_points

        try:
            interpolation_points = element.interpolation_points

            if len(interpolation_points) > 2:
                element_data = [
                    repr(extreme_points[0]),
                    repr(interpolation_points[0]),
                    " ... ",
                    repr(interpolation_points[-1]),
                    repr(extreme_points[1]),
                ]
            else:
                element_data.append(repr(extreme_points[0]))
                for point in interpolation_points:
                    element_data.append(f"{repr(point)}")
                element_data.append(repr(extreme_points[1]))
        except:
            for point in extreme_points:
                element_data.append(f"{repr(point)}")

        element_string = ",  ".join(element_data)

        end = f"\n  )"
        return begin + element_string + end

    def __str__(self) -> str:
        if len(self._elements) == 0:
            return "Elements()"

        begin = "Elements(\n  "

        middle = ",\n  ".join(
            [f"{self._string_for_element_info(element)}" for element in self._elements]
        )
        end = "\n)"
        return begin + middle + end

    def __getitem__(self, element_index: int) -> _ContainerElement:
        if element_index > self._index_next_element - 1:
            raise IndexError(
                f"The element index is greater than the number of elements {self._index_next_element -1}"
            )

        return self._elements[element_index]
