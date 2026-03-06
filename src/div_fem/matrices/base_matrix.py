import random
from typing import Literal, Self

from .operations import multiplication
from .lu_decomposition import lu_decomposition


class Matrix:
    _data: list[list[float]]

    rows: int
    columns: int
    type_of_print_specifier: Literal["scientific", "decimal", "integer"] = "decimal"

    def __init__(
        self,
        elements: list[list[float]] | None = None,
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
                raise ValueError(
                    "The number of elements must be valid. Some of your columns does have a different number of elements."
                )

            self.rows = len(elements)
            self.columns = len(elements[0])
            self._data = elements

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

    def _with_zeros(self, rows: int, columns: int) -> list[list[float]]:
        return [[0 for _ in range(rows)] for _ in range(columns)]

    def _with_random(self, rows: int, columns: int) -> list[list[float]]:
        return [[random.randint(-50, 100) for _ in range(rows)] for _ in range(columns)]

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

    @property
    def L(self) -> Matrix:
        l, _ = lu_decomposition(self._data)
        return Matrix(elements=l)

    @property
    def U(self) -> Matrix:
        _, u = lu_decomposition(self._data)
        return Matrix(elements=u)

    def __mul__(self, matrixB: Matrix | list[list[float]]) -> Matrix:
        if isinstance(matrixB, Matrix):
            matrixB = matrixB.get_list()

        return Matrix(multiplication(self._data, matrixB))

    def __str__(self) -> str:
        begin = "[\n  "
        if self.type_of_print_specifier == "scientific":
            rows_stringified = [
                ", ".join([f"{value:1.6E}" for value in row]) for row in self._data
            ]
        elif self.type_of_print_specifier == "decimal":
            rows_stringified = [
                ", ".join([f"{value:1.2f}" for value in row]) for row in self._data
            ]
        else:
            rows_stringified = [
                ", ".join([f"{int(value):>4d}" for value in row]) for row in self._data
            ]

        data_stringified = ",\n  ".join(["[ " + row + " ]" for row in rows_stringified])
        end = "\n]"

        return begin + data_stringified + end

    def get_list(self) -> list[list[float]]:
        return self._data
