from __future__ import annotations
import random
import numpy as np
from typing import Literal, Sequence, overload, Self
from .main_types import _MatrixDataType, _MatrixInputType, _VectorInputType
from .base_vector import Vector


class Matrix:
    _data: _MatrixDataType

    rows: int
    columns: int
    type_of_print_specifier: Literal["scientific", "decimal", "integer"] = "decimal"

    def __init__(
        self,
        elements: _MatrixInputType | None = None,
        rows: int | None = None,
        *,
        columns: int | None = None,
        random: bool = False,
        identity: bool = False,
    ) -> None:
        if not elements and not rows and not columns:
            raise ValueError(
                "A Matrix can be created passing your values or choosing a number of rows (square matrix will be considerable), rows and columns. If you declare rows or rows and columns, a matrix full of zeros will be created. You can create a matrix with random ints or the identity matrix, just passing this keyword arguments as True (just one of them)"
            )

        if (elements and rows) or (elements and rows and columns):
            raise ValueError("Choose creating a matrix with your values or your shape.")

        if random and identity:
            raise ValueError("Choose one of the basic types of matrix.")

        if elements:
            number_of_columns = [len(row) for row in elements]
            if max(number_of_columns) != min(number_of_columns):
                raise ValueError("The number of elements must be valid. Some of your columns does have a different number of elements.")

            self.rows = len(elements)
            self.columns = len(elements[0])
            self._data = [list(row) for row in elements]

        if rows:
            self.rows = rows
            if not columns:
                columns = self.rows

            self.columns = columns

            if random:
                self._data = self._with_random(self.rows, self.columns)
            elif identity:
                if self.rows != self.columns:
                    raise ValueError("The identity matrix must be square")
                self._data = self._with_diag_of_ones(self.rows)
            else:
                self._data = self._with_zeros(self.rows, self.columns)

    @property
    def shape(self) -> tuple:
        return (self.rows, self.columns)

    @property
    def T(self) -> Matrix:
        """Transposition of a matrix"""
        transpose_matrix = Matrix(rows=self.columns, columns=self.rows)

        for i in range(self.rows):
            for j in range(self.columns):
                transpose_matrix[j, i] = self._data[i][j]
        return transpose_matrix

    @property
    def norm(self) -> float:
        return np.sqrt(sum([self[i, j] ** 2 for i in range(self.rows) for j in range(self.columns)]))

    def get_list(self) -> _MatrixDataType:
        return self._data

    def print(self) -> None:
        print(self.__str__())

    def to_vector(self) -> Vector:
        if self.columns != 1 and self.rows != 1:
            raise ValueError(f"A matrix {self.rows} x {self.columns} can't be turned into a vector.")
        else:
            if self.rows == 1:
                return Vector(self._data[0])
            else:
                new_vector = Vector(rows=self.rows)
                for i in range(self.rows):
                    new_vector[i] = self[i, 0]
                return new_vector

    @staticmethod
    def zeros(m: int, n: int | None = None) -> Matrix:
        if not n:
            n = m
        return Matrix(rows=m, columns=n)

    @staticmethod
    def ones(m: int, n: int | None = None) -> Matrix:
        if not n:
            n = m

        return Matrix([[1 for _ in range(n)] for _ in range(m)])

    def _with_zeros(self, rows: int, columns: int) -> list[list[float]]:
        return [[0 for _ in range(columns)] for _ in range(rows)]

    def _with_random(self, rows: int, columns: int) -> list[list[float]]:
        return [[random.randint(-50, 100) for _ in range(columns)] for _ in range(rows)]

    def _with_diag_of_ones(self, rows: int) -> list[list[float]]:
        data = []

        for i in range(rows):
            row = []
            for j in range(rows):
                if i == j:
                    row.append(1)
                else:
                    row.append(0)
            data.append(row)

        return data

    def _extracting_data_as_string(self) -> list[str]:
        if self.type_of_print_specifier == "scientific":
            rows_stringified = [", ".join([f"{value:>10.3E}" for value in row]) for row in self._data]
        elif self.type_of_print_specifier == "decimal":
            rows_stringified = [", ".join([f"{value:12.6f}" for value in row]) for row in self._data]
        else:
            rows_stringified = [", ".join([f"{int(value):>4d}" for value in row]) for row in self._data]

        return rows_stringified

    def __repr__(self) -> str:
        begin = f"Matrix("
        data_stringified = ", ".join(["[" + row + "]" for row in self._extracting_data_as_string()])
        end = ")"
        return begin + data_stringified + end

    def __str__(self) -> str:
        begin = "[\n  "

        rows_stringified = self._extracting_data_as_string()

        data_stringified = ",\n  ".join(["[ " + row + " ]" for row in rows_stringified])
        end = "\n]"

        return begin + data_stringified + end

    def __add__(self, matrix: Matrix | _MatrixInputType) -> Matrix:
        if isinstance(matrix, Matrix):
            matrix = matrix.get_list()

        return Matrix((np.array(self._data) + np.array(matrix)).tolist())

    def __iadd__(self, matrix: Matrix | _MatrixInputType) -> Self:
        if not isinstance(matrix, Matrix):
            matrix = Matrix(matrix)

        if matrix.rows != self.rows:
            raise ValueError("To sum a matrix in place, the second matrix must have the same number of rows that the first.")

        if matrix.columns != self.columns:
            raise ValueError("To sum a matrix in place, the second matrix must have the same number of columns that the first.")

        for i in range(self.rows):
            for j in range(self.columns):
                self[i, j] += matrix[i, j]

        return self

    def __sub__(self, matrix: Matrix | _MatrixInputType) -> Matrix:
        if isinstance(matrix, Matrix):
            matrix = matrix.get_list()

        return Matrix((np.array(self._data) - np.array(matrix)).tolist())

    @overload
    def __mul__(self, second: Matrix | _MatrixInputType | int | float) -> Matrix: ...

    @overload
    def __mul__(self, second: Vector | _VectorInputType) -> Vector: ...

    def __mul__(
        self,
        second: Matrix | _MatrixInputType | Vector | _VectorInputType | int | float,
    ) -> Matrix | Vector:
        if isinstance(second, (int, float)):
            return Matrix((np.array(self._data) * second).tolist())
        elif isinstance(second, Matrix):
            second = second.get_list()
            return Matrix(np.matmul(np.array(self._data), np.array(second)).tolist())
        elif isinstance(second, Vector):
            second = second.get_list()
            return Vector(np.matmul(np.array(self._data), np.array(second)).tolist())
        else:
            if not isinstance(second, Sequence):
                second = second.get_list()
                return Vector(np.matmul(np.array(self._data), np.array(second)).tolist())

            if not isinstance(second[0], list):
                return Vector(np.matmul(np.array(self._data), np.array(second)).tolist())
            else:
                return Matrix(np.matmul(np.array(self._data), np.array(second)).tolist())

    def inv(self) -> Matrix:
        return Matrix(np.linalg.inv(self._data).tolist())

    @overload
    def __getitem__(self, idx: tuple[int, int]) -> float: ...

    @overload
    def __getitem__(self, idx: list[int] | tuple[list[int], list[int]]) -> Matrix: ...

    def __getitem__(
        self,
        idx: tuple[int, int] | tuple[list[int], list[int]] | list[int],
    ) -> Matrix | float:
        if not isinstance(idx, tuple):
            rows, columns = idx, idx

            matrix: list[list[float]] = []

            for r in rows:
                row = []
                for c in columns:
                    row.append(self._data[r][c])
                matrix.append(row)

            return Matrix(matrix)
        else:
            a, b = idx

            if isinstance(a, list):
                if isinstance(b, list):
                    matrix = []
                    for r in a:
                        rows = []
                        for c in b:
                            rows.append(self._data[r][c])
                        matrix.append(rows)
                    return Matrix(matrix)
                else:
                    raise ValueError(
                        "This matrix's values can be accessible in the ways:\n\t      "
                        + "A[2, 3] -> element on second row third column\n\t      "
                        + "A[[1, 4, 5]] -> all elements from first row and column, fourth row and column, and fifth row and column\n\t      "
                        + "A[[2, 3], [4, 5, 7]] -> all elements from second and third rows on fourth, fifth and seventh columns"
                    )
            else:
                if isinstance(b, list):
                    raise ValueError(
                        "This matrix's values can be accessible in the ways:\n\t      "
                        + "A[2, 3] -> element on second row third column\n\t      "
                        + "A[[1, 4, 5]] -> all elements from first row and column, fourth row and column, and fifth row and column\n\t      "
                        + "A[[2, 3], [4, 5, 7]] -> all elements from second and third rows on fourth, fifth and seventh columns"
                    )
                else:
                    return self._data[a][b]

    @overload
    def __setitem__(self, idx: tuple[int, int], values: float | int) -> None: ...

    @overload
    def __setitem__(
        self,
        idx: list[int] | tuple[list[int], list[int]],
        values: Matrix | _MatrixInputType,
    ) -> None: ...

    def __setitem__(
        self,
        idx: tuple[int, int] | tuple[list[int], list[int]] | list[int],
        values: Matrix | float | int | _MatrixInputType,
    ) -> None:
        # Case 1: Single element assignment M[i, j] = val
        if isinstance(idx, tuple) and isinstance(idx[0], int) and isinstance(idx[1], int):
            if not isinstance(values, (int, float)):
                raise ValueError("Single element assignment requires a scalar value.")
            self._data[idx[0]][idx[1]] = float(values)
            return

        target_rows: list[int]
        target_cols: list[int]

        if isinstance(idx, list):
            target_rows = idx
            target_cols = idx
        elif isinstance(idx, tuple) and isinstance(idx[0], list) and isinstance(idx[1], list):
            target_rows = idx[0]  # type: ignore
            target_cols = idx[1]  # type: ignore
        else:
            raise ValueError(
                "This matrix's values can be accessible in the ways:\n\t      "
                + "A[2, 3] -> element on second row third column\n\t      "
                + "A[[1, 4, 5]] -> all elements from first row and column, fourth row and column, and fifth row and column\n\t      "
                + "A[[2, 3], [4, 5, 7]] -> all elements from second and third rows on fourth, fifth and seventh columns"
            )

        if isinstance(values, Matrix):
            values = values.get_list()

        if not isinstance(values, list) or (values and not isinstance(values[0], list)):
            raise ValueError("The value for a list of indexes must be a valid Matrix or list of lists with the same size of the indexes")

        values_list: _MatrixInputType = values  # type: ignore

        if len(values_list) != len(target_rows):
            raise ValueError(f"Shape mismatch: expected {len(target_rows)} rows, got {len(values_list)}")

        for i, r in enumerate(target_rows):
            if len(values_list[i]) != len(target_cols):
                raise ValueError(f"Shape mismatch in row {i}: expected {len(target_cols)} columns, got {len(values_list[i])}")
            for j, c in enumerate(target_cols):
                self._data[r][c] = values_list[i][j]
