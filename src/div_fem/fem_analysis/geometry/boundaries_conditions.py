from __future__ import annotations
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from div_fem.fem_analysis.geometry.boundary_condition import BoundaryCondition


class BoundariesConditions:
    _initialized: bool = False
    _instance: BoundariesConditions | None = None

    _boundaries_cond: list[BoundaryCondition]

    _index_next_boundary_cond: int = 1

    def __new__(cls) -> Self:
        if cls._instance:
            raise ValueError(
                "The Boundaries Conditions object must be single per analysis."
            )

        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True

    @property
    def count(self) -> int:
        return self._index_next_boundary_cond - 1
