from typing import Literal

from div_fem.matrices.base_matrix import Matrix
from div_fem.fem_analysis.geometry.point import Point
from div_fem.matrices.base_vector import Vector
from .base_element import BaseElement


class Element2D(BaseElement):
    _type: Literal["bar", "beam", "frame"]
    _cosine_angles: tuple[float, float]
    _length: float

    def __init__(
        self,
        points: list[Point],
        degrees_of_freedom: list[int] | list[list[int]],
        type: Literal["bar", "beam", "frame"] = "bar",
    ) -> None:
        self._type = type
        super().__init__(points, degrees_of_freedom)
        self._length = self._calculating_length()
        self._cosine_angles = self._calculating_cosine_angles()

    @property
    def T(self) -> Matrix:
        """
        Rotation matrix for the 2D element to transform from local to global coordinates.
        """
        return Matrix(
            [
                [self._cosine_angles[0], self._cosine_angles[1], 0, 0],
                [0, 0, self._cosine_angles[0], self._cosine_angles[1]],
            ]
        )

    @property
    def type(self) -> Literal["bar", "beam", "frame"]:
        return self._type

    @property
    def length(self) -> float:
        return self._length

    def _calculating_length(self) -> float:
        return (Vector(self._points[1]) - Vector(self._points[0])).norm

    def _calculating_cosine_angles(self) -> tuple[float, float]:
        c = (self._points[1][0] - self._points[0][0]) / self._length
        s = (self._points[1][1] - self._points[0][1]) / self._length

        return (c, s)
