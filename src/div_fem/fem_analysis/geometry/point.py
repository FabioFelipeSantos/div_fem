from __future__ import annotations
from typing import TYPE_CHECKING, overload
import numpy as np

from .main_geometry_types import _PointInputData, _PointData

from div_fem.utils.descriptors.descriptor_private_name import (
    DescriptorBaseClass,
    PrivateName,
)


class PointIndex(PrivateName):

    def __get__(self, obj: Point, objtype=None) -> int:
        value = getattr(obj, self.private_name, None)

        if not value:
            raise AttributeError(
                "The Point index can only be accessed after it being in Points class"
            )

        return value

    def __set__(self, obj: Point, value: int) -> None:
        if value < 0:
            raise ValueError(
                f"A index for point must be greater than 0. Received {value}"
            )
        setattr(obj, self.private_name, value)


class DOFNumbers(PrivateName):

    def __get__(self, obj: Point, objtype=None) -> list[int]:
        value = getattr(obj, self.private_name, None)

        if not value:
            raise AttributeError(
                "The DOFNumbers can only be accessed after the point being in the Point class with an Structural Analysis class instantiated"
            )

        return value

    def __set__(self, obj: Point, value: list[int]) -> None:
        setattr(obj, self.private_name, value)


class Point:

    _coord: _PointData
    _dimension: int

    dof_numbers = DOFNumbers()
    index = PointIndex()
    dof_per_node = DescriptorBaseClass[int]()

    def __init__(self, coordinates: _PointInputData) -> None:
        self._coord = list(coordinates)
        self._dimension = len(coordinates)

    @property
    def norm(self) -> float:
        return np.sqrt(sum([x**2 for x in self._coord]))

    @property
    def dimension(self) -> int:
        return self._dimension

    def get_list(self) -> _PointData:
        return self._coord

    def print(self) -> None:
        print(self.__repr__())

    def __getitem__(self, idx: int) -> float:
        if idx > self._dimension - 1:
            raise IndexError(
                f"Received {idx} as index for a point with {self._dimension} coordinates."
            )
        return self._coord[idx]

    @overload
    def __setitem__(self, idx: list[int], values: _PointInputData) -> None: ...

    @overload
    def __setitem__(self, idx: int, values: float) -> None: ...

    def __setitem__(
        self, idx: list[int] | int, values: _PointInputData | float
    ) -> None:
        if isinstance(idx, list):
            if len(idx) > self._dimension:
                raise ValueError(
                    f"The indexes list has more values than the dimension {self._dimension} of the point."
                )

            if not isinstance(values, list):
                raise ValueError(
                    "To set more than one coordinate for the point, provide a list of values for each coordinate."
                )

            if len(values) > len(idx):
                raise ValueError(
                    "The provided values have more values than the list indexes."
                )

            for i in idx:
                self._coord[i] = values[i]
        else:
            if idx > self._dimension:
                raise IndexError(
                    f"The provided index {idx} is greater than dimension {self._dimension} of the point."
                )

            if not isinstance(values, float):
                raise ValueError("For just one coordinate provide one value for it.")

            self._coord[idx] = values

    def __len__(self) -> int:
        return self._dimension

    def __str__(self) -> str:
        return "(" + self._extracting_string_from_data() + ")"

    def __repr__(self) -> str:
        return "Point(" + self._extracting_string_from_data() + ")"

    def _extracting_string_from_data(self) -> str:
        return ", ".join([f"{value:5.2f}" for value in self._coord])
