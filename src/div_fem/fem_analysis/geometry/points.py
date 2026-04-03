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
    _index_next_point: int = 1

    def __new__(cls) -> Self:
        if cls._instance:
            raise ValueError("The Points object must be single per analysis.")

        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True

    @overload
    def __call__(self, elements: int) -> Point:
        """
        Calling the Points class with the element as int value will return the point associated at that int nodal index. The indexation is based on 0 array index Python, but will must provide the nodal index wanted in normal indexation.

        For example:
            A point in the structure labeled with the number 10 will have the number 9 as indexation in Points class. Therefore, calling Points(10) will return points_list[9].
        """
        pass

    @overload
    def __call__(self, elements: Point | list[Point]) -> None:
        """
        Calling the Points class with the element as Point instance will add the point to the class and will calculate your degrees of freedom list based in the value of number of degree of freedom per node, as declared when Points was instantiated.

        For example:
            If we call points(Point(1, 1)) after adding 9 points in a 2 degree of freedom per node analysis, the point (1, 1) will be the 10th node in index 9 with degrees as [18, 19]
        """
        pass

    def __call__(self, elements: int | Point | list[Point]) -> Point | None:
        if isinstance(elements, int):
            return self.point(elements)
        else:
            self.add(elements)
            return

    @property
    def number_of_points(self) -> int:
        return len(self._points)

    def point(self, nodal_index: int) -> Point:
        return self._points[nodal_index - 1]

    def add(self, points: Point | list[Point]) -> None:
        if isinstance(points, list):
            for point in points:
                self._add_new_point(point)
        else:
            self._add_new_point(points)

        return

    def _add_new_point(self, point: Point) -> None:
        self._points.append(point)
        point.set_point_index(self._index_next_point)
        self._index_next_point += 1

    def __str__(self) -> str:
        if len(self._points) == 1:
            return "Points()"

        begin = "Points(\n    "

        middle = ",\n    ".join(
            [f"Point[{idx}]{str(point)}" for idx, point in enumerate(self._points)]
        )
        end = "\n)"
        return begin + middle + end
