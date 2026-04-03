from __future__ import annotations
from typing import TYPE_CHECKING, Self


if TYPE_CHECKING:
    from div_fem.fem_analysis.geometry.points import Points


class DegreeOfFreedom:
    # def __set__(self, value: int) ->
    pass


class Structural2DAnalysis:
    _initialized: bool = False
    _instance: Structural2DAnalysis | None = None

    _points: "Points | None" = None

    def __new__(cls) -> Self:
        if cls._instance:
            raise ValueError("The Structural2DAnalysis class is single to the analysis.")

        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True
