from typing import Sequence

from div_fem.fem_analysis.geometry.point import Point

_VectorDataType = list[float]
_MatrixDataType = list[list[float]]

_VectorInputType = Sequence[float] | Point
_MatrixInputType = Sequence[Sequence[float]]
