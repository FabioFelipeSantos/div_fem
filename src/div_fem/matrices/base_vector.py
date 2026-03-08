from __future__ import annotations
import random
from typing import Literal, overload, Any, cast

import numpy as np
from .main_types import _VectorDataType
from .base_matrix import Matrix


class Vector:
    rows: int
    _data: _VectorDataType
    type_of_print_specifier: Literal["scientific", "decimal", "integer"] = "decimal"

    def __init__(
        self,
        elements: _VectorDataType | None = None,
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
            raise ValueError(
                "Choose one of the basic types of vector: random or unit direction vector."
            )

        if elements:
            self.rows = len(elements)
            self._data = elements

        if rows:
            self.rows = rows

            if random:
                self._data = self._with_random(self.rows)
            elif unity_direction_vector_dim:
                self._data = _with_one_in_dimension_n(
                    self.rows, unity_direction_vector_dim
                )
            else:
                self._data = self._with_zeros(self.rows)

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

    def get_list(self) -> _VectorDataType:
        return self._data

    def print(self) -> None:
        print(self.__str__())

    @overload
    def dot(self, other: Vector) -> float: ...

    @overload
    def dot(self, other: _VectorDataType) -> float: ...

    def dot(self, other: Vector | _VectorDataType) -> float:
        if isinstance(other, list):
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
            rows_stringified = [f"{value:1.2f}" for value in self._data]
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

    def __add__(self, vector: Vector | _VectorDataType) -> Vector:
        if isinstance(vector, Vector):
            vector = vector.get_list()

        return Vector((np.array(self._data) + np.array(vector)).tolist())

    def __sub__(self, vector: Vector | _VectorDataType) -> Vector:
        if isinstance(vector, Vector):
            vector = vector.get_list()

        return Vector((np.array(self._data) - np.array(vector)).tolist())

    def __len__(self) -> int:
        return self.rows

    @overload
    def __mul__(self, other: Matrix) -> Matrix: ...

    @overload
    def __mul__(self, other: list[list[float]]) -> Matrix: ...

    @overload
    def __mul__(self, other: float) -> Vector: ...

    def __mul__(self, other: Matrix | list[list[float]] | float) -> Matrix | Vector:
        if isinstance(other, (float, int)):
            return Vector((np.array(self._data) * other).tolist())

        if isinstance(other, list):
            other = Matrix(other)

        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Shape mismatch error for matrix. Hope to get {self.shape[1]}, received {other.shape[0]}."
            )

        matrix_multiplication = Matrix(rows=self.shape[0], columns=other.shape[1])
        for r in range(self.shape[0]):
            for c in range(other.shape[1]):
                matrix_multiplication[r, c] = self._data[r] * other[0, c]
        return matrix_multiplication

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
        values: Vector | _VectorDataType,
    ) -> None: ...

    def __setitem__(
        self,
        idx: int | list[int],
        values: Vector | float | int | _VectorDataType,
    ) -> None:
        if isinstance(idx, int):
            if not isinstance(values, (float, int)):
                raise ValueError("Single element assignment requires a scalar value.")
            cast(list[Any], self._data)[idx] = values
            return

        if isinstance(values, Vector):
            values = values.get_list()

        if not isinstance(values, list):
            raise ValueError(
                "The value for a list of indexes must be a valid Vector or list of scalars with the same size of the indexes"
            )

        if len(values) != len(idx):
            raise ValueError(
                f"Shape mismatch: expected {len(idx)} rows, got {len(values)}"
            )

        for i, r in enumerate(idx):
            cast(list[Any], self._data)[r] = values[i]

    @staticmethod
    def zeros(m: int) -> Vector:
        return Vector(rows=m)

    @staticmethod
    def ones(m: int) -> Vector:
        return Vector([1 for _ in range(m)])

    @staticmethod
    def unit_direction_vector(rows: int, unity_direction_vector_dim: int) -> Vector:
        return Vector(_with_one_in_dimension_n(rows, unity_direction_vector_dim))

    @staticmethod
    def vector_norm(vector: Vector | _VectorDataType) -> float:
        if isinstance(vector, Vector):
            vector = vector.get_list()
        return _calculating_norm(vector)

    @staticmethod
    def vecdot(
        vectorA: Vector | _VectorDataType, vectorB: Vector | _VectorDataType
    ) -> float:
        if isinstance(vectorA, list):
            vectorA = Vector(vectorA)

        if isinstance(vectorB, list):
            vectorB = Vector(vectorB)

        return _calculating_dot(vectorA, vectorB)


def _with_one_in_dimension_n(
    rows: int, unity_direction_vector_dim: int
) -> _VectorDataType:
    data: _VectorDataType = []

    for i in range(rows):
        if i == unity_direction_vector_dim:
            data.append(1)
        else:
            data.append(0)

    return data


def _calculating_norm(vector: _VectorDataType) -> float:
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
