from __future__ import annotations
from typing import Self, NamedTuple, overload

from div_fem.fem_analysis.geometry.point import Point


class PointInfo(NamedTuple):
    nodal_index_number: int
    point: Point
    degrees_of_freedom: list[int]


class Points:
    _initialized: bool = False
    _instance: Points | None = None

    _points: list[Point] = []
    _degrees_of_freedom: list[list[int]] = []

    _index_next_point: int = 1

    _number_of_degree_of_freedom_per_node: int
    _range_number_dof: range

    def __new__(cls, number_of_degree_of_freedom_per_node: int) -> Self:
        if cls._instance:
            raise ValueError("The Points object must be single per analysis.")

        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, number_of_degree_of_freedom_per_node: int) -> None:
        if self._initialized:
            return

        self._initialized = True
        self._number_of_degree_of_freedom_per_node = (
            number_of_degree_of_freedom_per_node
        )
        self._range_number_dof = range(self._number_of_degree_of_freedom_per_node)

    @overload
    def __call__(self, element: int) -> Point:
        """
        Calling the Points class with the element as int value will return the point associated at that int nodal index. The indexation is based on 0 array index Python, but will must provide the nodal index wanted in normal indexation.

        For example:
            A point in the structure labeled with the number 10 will have the number 9 as indexation in Points class. Therefore, calling Points(10) will return points_list[9].
        """
        pass

    @overload
    def __call__(self, element: Point) -> None:
        """
        Calling the Points class with the element as Point instance will add the point to the class and will calculate your degrees of freedom list based in the value of number of degree of freedom per node, as declared when Points was instantiated.

        For example:
            If we call points(Point(1, 1)) after adding 9 points in a 2 degree of freedom per node analysis, the point (1, 1) will be the 10th node in index 9 with degrees as [18, 19]
        """
        pass

    def __call__(self, element: int | Point) -> Point | None:
        if isinstance(element, int):
            return self.point(element)

        if isinstance(element, Point):
            self.add(element)
            return

    @property
    def number_of_points(self) -> int:
        return self._index_next_point - 1

    @property
    def degrees_of_freedom(self) -> list[list[int]]:
        return self._degrees_of_freedom

    def degree_of_freedom(self, nodal_index: int) -> list[int]:
        return self._degrees_of_freedom[nodal_index - 1]

    def point(self, nodal_index: int) -> Point:
        return self._points[nodal_index - 1]

    def point_info(self, nodal_index: int) -> PointInfo:
        return PointInfo(
            nodal_index - 1,
            self.point(nodal_index),
            self.degree_of_freedom(nodal_index),
        )

    def add(self, point: Point) -> None:
        self._points.append(point)

        self._degrees_of_freedom.append(
            [
                self._number_of_degree_of_freedom_per_node
                * (self._index_next_point - 1)
                + dof_index
                for dof_index in self._range_number_dof
            ]
        )
        self._index_next_point += 1
        return

    def __str__(self) -> str:
        if self._index_next_point == 1:
            return "Points()"

        begin = "Points(\n    "

        middle = ",\n    ".join(
            [f"Point[{idx}]{str(point)}" for idx, point in enumerate(self._points)]
        )
        end = "\n)"
        return begin + middle + end
