from __future__ import annotations
from typing import TYPE_CHECKING, Self, overload

if TYPE_CHECKING:
    from div_fem.fem_analysis.geometry.boundary_condition import BoundaryCondition


class BoundariesConditions:
    _initialized: bool = False
    _instance: BoundariesConditions | None = None

    boundaries_cond: list[BoundaryCondition] = []

    _index_next_boundary_cond: int = 1

    def __new__(cls) -> Self:
        if cls._instance:
            raise ValueError("The Boundaries Conditions object must be single per analysis.")

        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True

    @property
    def count(self) -> int:
        return self._index_next_boundary_cond - 1

    @overload
    def __call__(self, boundaries: int) -> BoundaryCondition:
        """
        Calling the BoundariesConditions class with the boundaries as int value will return the Boundary Condition associated at that index. The indexation is based on 0 array index Python.
        """
        pass

    @overload
    def __call__(self, boundaries: BoundaryCondition | list[BoundaryCondition]) -> None:
        """
        Calling the BoundariesConditions class with the boundaries as BoundaryCondition instance or a list of BoundaryCondition instances, will add the conditions to the class.
        """
        pass

    def __call__(self, boundaries: int | BoundaryCondition | list[BoundaryCondition]) -> BoundaryCondition | None:
        if isinstance(boundaries, int):
            return self.boundaries_cond[boundaries]
        else:
            self.add(boundaries)
            return

    def add(self, boundaries: BoundaryCondition | list[BoundaryCondition]) -> None:
        if isinstance(boundaries, list):
            for condition in boundaries:
                self._add_new_condition(condition)
        else:
            self._add_new_condition(boundaries)

        return

    def _add_new_condition(self, condition: BoundaryCondition) -> None:
        self.boundaries_cond.append(condition)
        condition.index = self._index_next_boundary_cond
        self._index_next_boundary_cond += 1

    def __str__(self) -> str:
        begin = "Conditions(\n  "

        middle = ",\n  ".join([f"{condition.index}: {str(condition)}" for condition in self.boundaries_cond])

        end = "\n)"

        return begin + middle + end
