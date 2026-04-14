from __future__ import annotations
import random
from typing import Literal, Self, overload, TYPE_CHECKING

import numpy as np

from div_fem.fem_analysis.geometry.point import Point
from .main_types import _VectorDataType, _VectorInputType, _MatrixInputType

if TYPE_CHECKING:
    from .base_matrix import Matrix


class Vector:
    rows: int
    _data: _VectorDataType
    type_of_print_specifier: Literal["scientific", "decimal", "integer"] = "decimal"

    def __init__(
        self,
        elements: _VectorInputType | None = None,
        rows: int | None = None,
        *,
        random: bool = False,
        unity_direction_vector_dim: int | None = None,
    ) -> None:
        if not elements and not rows:
            raise ValueError(
                "A Vector can be created passing your values or choosing a number of rows (elements). If you declare rows, a vector full of zeros will be created. You can create a vector with random ints, just passing the random keyword argument as True. If you want a vector representing a nth unit direction vector in direction n_i, pass unity_vector_dim with the index in that unit must be present (index starting at 0). Don't pass random=True with unity_vector_dim, this will raise a ValueError."
            )

        if elements and rows:
            raise ValueError("Choose creating a vector with your values or your shape.")

        if random and unity_direction_vector_dim:
            raise ValueError("Choose one of the basic types of vector: random or unit direction vector.")

        if elements:
            if isinstance(elements, Point):
                self.rows = elements.dimension
                self._data = elements.get_list()
            else:
                self.rows = len(elements)
                self._data = list(elements)

            self._enumeration = enumerate(self._data)
            return

        if rows:
            self.rows = rows

            if random:
                self._data = self._with_random(self.rows)
            elif unity_direction_vector_dim:
                self._data = _with_one_in_dimension_n(self.rows, unity_direction_vector_dim)
            else:
                self._data = self._with_zeros(self.rows)

            return

    @property
    def shape(self) -> tuple:
        return (self.rows, 1)

    @property
    def T(self) -> Matrix:
        transpose_vector = Matrix(rows=1, columns=self.rows)

        for i in range(self.rows):
            transpose_vector[0, i] = self._data[i]
        return transpose_vector

    @property
    def norm(self) -> float:
        return _calculating_norm(self._data)

    @property
    def enumeration(self) -> enumerate[float]:
        return enumerate(self._data)

    def index(self, value: float) -> int:
        return self._data.index(value)

    def get_list(self) -> _VectorDataType:
        return self._data

    def print(self) -> None:
        print(self.__str__())

    @overload
    def dot(self, other: Vector) -> float: ...

    @overload
    def dot(self, other: _VectorInputType) -> float: ...

    def dot(self, other: Vector | _VectorInputType) -> float:
        if not isinstance(other, Vector):
            other = Vector(other)

        return _calculating_dot(self, other)

    def _with_zeros(self, rows: int) -> list[float]:
        return [0 for _ in range(rows)]

    def _with_random(self, rows: int) -> list[float]:
        return [random.randint(-50, 100) for _ in range(rows)]

    def _extracting_data_as_string(self) -> list[str]:
        if self.type_of_print_specifier == "scientific":
            rows_stringified = [f"{value:1.6E}" for value in self._data]
        elif self.type_of_print_specifier == "decimal":
            rows_stringified = [f"{value:1.4f}" for value in self._data]
        else:
            rows_stringified = [f"{int(value):>4d}" for value in self._data]

        return rows_stringified

    def __repr__(self) -> str:
        begin = f"Vector("
        end = ")"
        return begin + ", ".join(self._extracting_data_as_string()) + end

    def __str__(self) -> str:
        begin = "[\n  "

        vector_stringified = self._extracting_data_as_string()

        data_stringified = ",\n  ".join([" " + row for row in vector_stringified])
        end = "\n]"

        return begin + data_stringified + end

    def __add__(self, vector: Vector | _VectorInputType) -> Vector:
        if isinstance(vector, Vector):
            vector = vector.get_list()

        return Vector((np.array(self._data) + np.array(vector)).tolist())

    def __iadd__(self, vector: Vector | _VectorInputType) -> Self:
        if not isinstance(vector, Vector):
            vector = Vector(vector)

        if vector.rows != self.rows:
            raise ValueError("To sum a vector in place, the second vector must have the same number of rows that the first.")

        for i in range(self.rows):
            self[i] += vector[i]

        return self

    def __sub__(self, vector: Vector | _VectorInputType) -> Vector:
        if isinstance(vector, Vector):
            vector = vector.get_list()

        return Vector((np.array(self._data) - np.array(vector)).tolist())

    def __len__(self) -> int:
        return self.rows

    @overload
    def __mul__(self, other: Matrix) -> Matrix: ...

    @overload
    def __mul__(self, other: _MatrixInputType) -> Matrix: ...

    @overload
    def __mul__(self, other: float) -> Vector: ...

    def __mul__(self, other: Matrix | _MatrixInputType | float) -> Matrix | Vector:
        if isinstance(other, (float, int)):
            return Vector((np.array(self._data) * other).tolist())

        if not isinstance(other, Matrix):
            other = Matrix(other)

        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Shape mismatch error for matrix. Hope to get {self.shape[1]}, received {other.shape[0]}.")

        matrix_multiplication = Matrix(rows=self.shape[0], columns=other.shape[1])
        for r in range(self.shape[0]):
            for c in range(other.shape[1]):
                matrix_multiplication[r, c] = self._data[r] * other[0, c]
        return matrix_multiplication

    def __rmul__(self, other: float) -> Vector:
        return Vector((np.array(self._data) * other).tolist())

    @overload
    def __getitem__(self, idx: int) -> float: ...

    @overload
    def __getitem__(self, idx: list[int]) -> Vector: ...

    def __getitem__(
        self,
        idx: int | list[int],
    ) -> Vector | float:

        if not isinstance(idx, int):
            vector: list[float] = []

            for r in idx:
                vector.append(self._data[r])

            return Vector(vector)
        else:
            return self._data[idx]

    @overload
    def __setitem__(self, idx: int, values: float | int) -> None: ...

    @overload
    def __setitem__(
        self,
        idx: list[int],
        values: Vector | _VectorInputType,
    ) -> None: ...

    def __setitem__(
        self,
        idx: int | list[int],
        values: Vector | float | int | _VectorInputType,
    ) -> None:
        if isinstance(idx, int):
            if not isinstance(values, (float, int)):
                raise ValueError("Single element assignment requires a scalar value.")
            self._data[idx] = values
            return

        if isinstance(values, Vector):
            values = values.get_list()

        if not isinstance(values, list):
            raise ValueError(
                "The value for a list of indexes must be a valid Vector or list of scalars with the same size of the indexes"
            )

        if len(values) != len(idx):
            raise ValueError(f"Shape mismatch: expected {len(idx)} rows, got {len(values)}")

        for i, r in enumerate(idx):
            self._data[r] = values[i]

    @staticmethod
    def zeros(m: int) -> Vector:
        return Vector(rows=m)

    @staticmethod
    def ones(m: int) -> Vector:
        return Vector([1 for _ in range(m)])

    @staticmethod
    def random(m: int) -> Vector:
        return Vector(rows=m, random=True)

    @staticmethod
    def unit_direction_vector(rows: int, unity_direction_vector_dim: int) -> Vector:
        return Vector(_with_one_in_dimension_n(rows, unity_direction_vector_dim))

    @staticmethod
    def vector_norm(vector: Vector | _VectorInputType) -> float:
        if isinstance(vector, Vector):
            vector = vector.get_list()
        return _calculating_norm(vector)

    @staticmethod
    def vecdot(vectorA: Vector | _VectorInputType, vectorB: Vector | _VectorInputType) -> float:
        if not isinstance(vectorA, Vector):
            vectorA = Vector(vectorA)

        if not isinstance(vectorB, Vector):
            vectorB = Vector(vectorB)

        return _calculating_dot(vectorA, vectorB)


def _with_one_in_dimension_n(rows: int, unity_direction_vector_dim: int) -> _VectorDataType:
    data: _VectorDataType = []

    for i in range(rows):
        if i == unity_direction_vector_dim:
            data.append(1)
        else:
            data.append(0)

    return data


def _calculating_norm(vector: _VectorInputType) -> float:
    return np.sqrt(sum([x**2 for x in vector]))


def _calculating_dot(vectorA: Vector, vectorB: Vector) -> float:
    if vectorA.rows != vectorB.rows:
        raise ValueError(
            f"Shape mismatch error. For the dot product must vectors must be the same size. Rows vectorA: {vectorA.rows}, Rows vectorB: {vectorB.rows}."
        )

    sum: float = 0.0
    for i in range(vectorA.rows):
        sum += vectorA[i] * vectorB[i]
    return sum
