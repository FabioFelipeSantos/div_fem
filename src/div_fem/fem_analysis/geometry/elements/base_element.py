from div_fem.fem_analysis.geometry.elements.shape_functions_interface import (
    ShapeFunctions,
)
from div_fem.fem_analysis.geometry.point import Point


class BaseElement:
    _points: list[Point]
    _number_of_points_for_interpolation: int
    _degrees_of_freedom: list[int] | list[list[int]]
    _total_element_degree_of_freedom: int
    _shape_functions: ShapeFunctions

    def __init__(
        self,
        points: list[Point],
        number_of_points_for_interpolation: int,
        degrees_of_freedom: list[int] | list[list[int]],
    ) -> None:
        if len(points) < 2:
            raise ValueError("An element can only be created with 2 or more points.")

        if len(degrees_of_freedom) < 2:
            raise ValueError(
                "The degrees_of_freedom for the element must have two or more degree of freedom numbers."
            )

        self._points = points
        self._number_of_points_for_interpolation = number_of_points_for_interpolation
        self._degrees_of_freedom = degrees_of_freedom

    @property
    def element_degree_of_freedom(self) -> int:
        """The number of total degree of freedom of the element. Useful for making the element stiffness matrix and forces vector."""
        return self._total_element_degree_of_freedom

    @property
    def number_points(self) -> int:
        """The number of points used in interpolation functions this element."""
        return self._number_of_points_for_interpolation

    @property
    def shape_functions(self) -> ShapeFunctions:
        return self._shape_functions

    def __len__(self) -> int:
        return len(self._points)

    def print(self, idx: int | None = None) -> None:
        """
        This function prints the element information. If a argument idx is passed, so the info of the point is provided as (Point, Degrees of Freedom)
        """
        if not idx:
            print("\n" + self.__str__() + "\n")
        else:
            print(f"\n{self[idx]}\n")

    def __str__(self) -> str:
        begin = "Element:\n  "
        element_data: list[str] = []

        for idx, point in enumerate(self._points):
            element_data.append(
                f"{idx + 1}: {repr(point)} -> {self._degrees_of_freedom[idx]},"
            )

        element_string = "\n  ".join(element_data)

        return begin + element_string

    def __repr__(self) -> str:
        return f"Element({", ".join([repr(point) for point in self._points])})"

    def __getitem__(self, idx: int) -> tuple[Point, int | list[int]]:
        """
        The index is the point number, starting at 0, used in the list to create this element.

        Returns:
            tuple(Point, int | list[int]): A tuple with the point and the degree of freedom of the point
        """
        if idx > self.number_points - 1:
            raise IndexError(
                f"The element doesn't have more than {self.number_points} points. The index count starts at zero."
            )

        return (self._points[idx], self._degrees_of_freedom[idx])
