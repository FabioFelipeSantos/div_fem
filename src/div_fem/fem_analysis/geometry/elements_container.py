from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Self, Union, overload

from div_fem.fem_analysis.geometry.elements.element_load_interface import (
    ElementLoadInterface,
)
from div_fem.fem_analysis.geometry.elements.shape_functions_interface import (
    ShapeFunctions,
)
from div_fem.fem_analysis.geometry.elements_interface import ElementInterface

if TYPE_CHECKING:
    from div_fem.fem_analysis.geometry.point import Point


class Elements:
    _initialized: bool = False
    _instance: Elements | None = None

    _elements: list[
        ElementInterface[
            str,
            Mapping[str, Any],
            Mapping[str, Any],
            ShapeFunctions,
            ElementLoadInterface[
                str,
                str,
                Any,
                Union[Mapping[str, Any], float],
                Union[float, Point, None],
            ],
        ]
    ] = []

    _degrees_of_freedom: list[list[int]] = []

    _index_next_element: int = 1

    _number_of_interpolation_points: int
    _number_of_degree_of_freedom_per_node: int
    _range_number_dof: range

    def __new__(
        cls,
        number_of_interpolation_points: int,
        number_of_degree_of_freedom_per_element: int,
    ) -> Self:
        if cls._instance:
            raise ValueError("The Elements object must be single per analysis.")

        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        number_of_interpolation_points: int,
        number_of_degree_of_freedom_per_node: int,
    ) -> None:
        if self._initialized:
            return

        self._initialized = True
        self._number_of_interpolation_points = number_of_interpolation_points
        self._number_of_degree_of_freedom_per_node = (
            number_of_degree_of_freedom_per_node
        )
        self._range_number_dof = range(self._number_of_degree_of_freedom_per_node)

    @property
    def number_of_elements(self) -> int:
        return len(self._elements)

    @property
    def number_of_interpolation_points(self) -> int:
        """The number of points used in interpolation functions this element."""
        return self._number_of_interpolation_points

    @property
    def total_degree_of_freedom(self) -> int:
        """The number of total degree of freedom of the element. Useful for making the element stiffness matrix and forces vector."""
        return (
            self._number_of_interpolation_points
            * self._number_of_degree_of_freedom_per_node
        )

    @property
    def degrees_of_freedom(self) -> list[list[int]]:
        return self._degrees_of_freedom

    def degree_of_freedom(self, element_index: int) -> list[int]:
        return self._degrees_of_freedom[element_index - 1]

    @overload
    def __call__(self, elements_or_index: int) -> ElementInterface: ...

    @overload
    def __call__(
        self, elements_or_index: ElementInterface | list[ElementInterface]
    ) -> None: ...

    def __call__(
        self, elements_or_index: int | ElementInterface | list[ElementInterface]
    ) -> ElementInterface | None:
        if isinstance(elements_or_index, int):
            return self._elements[elements_or_index - 1]
        else:
            self.add(elements_or_index)

    def add(self, elements: ElementInterface | list[ElementInterface]) -> None:
        if isinstance(elements, list):
            for element in elements:
                self._adding_one_element(element)
        else:
            self._adding_one_element(elements)

        return

    def _adding_one_element(self, element: ElementInterface) -> None:
        element.set_elements_container(self)
        element.set_element_index(self._index_next_element)
        self._elements.append(element)
        self._degrees_of_freedom.append(self._calculate_degree_of_freedom(len(element)))
        self._index_next_element += 1

    def _calculate_degree_of_freedom(self, number_of_points: int) -> list[int]:
        """This function will must be moved to the element and here is just a reference for linear 2D elements"""

        dof: list[int] = []

        element_number = self._index_next_element
        number_interpolation = self._number_of_interpolation_points
        r = range(
            number_interpolation * (element_number - 1),
            number_interpolation * element_number,
        )

        for n in r:
            for d in self._range_number_dof:
                dof.append(self._number_of_degree_of_freedom_per_node * n + d)

        return dof

    def dof_string(self, index: int) -> str:
        element_dof = self._degrees_of_freedom[index - 1]
        n = self._number_of_degree_of_freedom_per_node
        data = [element_dof[n * i : n * (i + 1)] for i in range(n)]
        return str(data)

    def _string_for_element_info(self, element: ElementInterface) -> str:
        begin = f"Element[{element.index:>3d}]:(\n    "
        element_data: list[str] = []

        for idx, point in enumerate(element.points):
            element_data.append(f"{idx + 1}: {repr(point)}")

        element_string = ",  ".join(element_data)

        end = f"\n    Number of interpolation points = {self._number_of_interpolation_points}\n    DOF = {self.dof_string(element.index)}\n  )"
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

    def __getitem__(self, element_index: int) -> ElementInterface:
        if element_index < 0:
            raise IndexError(
                "There's not negative indexation for the Elements container."
            )

        if element_index == 0:
            raise IndexError(
                "The element number must be usual non 0 indexation. Just inside the class that 0 indexation is used."
            )

        if element_index > self._index_next_element - 1:
            raise IndexError(
                f"The element index is greater than the number of elements {self._index_next_element -1}"
            )

        return self._elements[element_index - 1]
