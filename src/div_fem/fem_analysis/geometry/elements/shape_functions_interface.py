from abc import ABC, abstractmethod
from typing import Literal, overload

from div_fem.matrices.base_matrix import Matrix
from div_fem.matrices.base_vector import Vector


class ShapeFunctions(ABC):
    _number_of_points: int
    _interpolation_points: Vector
    _barycentric_weights: Vector
    _nodal_inclination: Vector | None = None
    _jacobian: float

    @overload
    @abstractmethod
    def value(self, xi: float) -> Matrix: ...

    @overload
    @abstractmethod
    def value(self, xi: float, index: int) -> float | list[float]: ...

    @abstractmethod
    def value(
        self, xi: float, index: int | None = None
    ) -> Matrix | float | list[float]: ...

    @overload
    @abstractmethod
    def derivative(
        self, diff_order: int, xi: float, *, is_for_stiffness_matrix: bool = False
    ) -> Matrix: ...

    @overload
    @abstractmethod
    def derivative(
        self,
        diff_order: int,
        xi: float,
        index: int,
        *,
        is_for_stiffness_matrix: bool = False,
    ) -> float: ...

    @overload
    @abstractmethod
    def derivative(
        self,
        diff_order: int,
        xi: float,
        index: list[int],
        *,
        is_for_stiffness_matrix: bool = False,
    ) -> list[float]: ...

    @abstractmethod
    def derivative(
        self,
        diff_order: int,
        xi: float,
        index: int | list | None = None,
        *,
        is_for_stiffness_matrix: bool = False,
    ) -> Matrix | float | list[float]: ...

    @property
    @abstractmethod
    def number_of_points(self) -> int: ...

    @property
    @abstractmethod
    def interpolation_points(self) -> Vector: ...

    @property
    @abstractmethod
    def barycentric_weights(self) -> Vector: ...

    @property
    @abstractmethod
    def nodal_inclination(self) -> Vector | None: ...

    @property
    @abstractmethod
    def jacobian(self) -> float:
        return self._jacobian

    def __str__(self) -> str:
        return f"\nShapeFunctions(\n    number_of_points={self._number_of_points},\n    points={repr(self._interpolation_points)}\n)\n"
