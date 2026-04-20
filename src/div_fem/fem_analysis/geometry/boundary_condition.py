from typing import TYPE_CHECKING, Literal, Mapping, NotRequired, TypedDict, overload

if TYPE_CHECKING:
    from div_fem.fem_analysis.geometry.point import Point


# class BoundaryCondition2DInfo(TypedDict):
#     x: NotRequired[float]
#     y: NotRequired[float]
#     moment: NotRequired[float]
#     rotation: NotRequired[float]

_BoundaryInfoKeys = Literal["x", "y", "moment"]
BOUNDARY_NAMES = ["x", "y", "moment"]

_BoundaryCondition2DInfo = Mapping[_BoundaryInfoKeys, float | None]


class BoundaryCondition:
    _point: "Point"
    _boundary_info: _BoundaryCondition2DInfo
    _rotation: None | float

    index: int

    def __init__(
        self,
        point: Point,
        boundary_info: _BoundaryCondition2DInfo,
        rotation: float | None = None,
    ) -> None:
        self._point = point
        x_condition = boundary_info.get("x", None)
        y_condition = boundary_info.get("y", None)
        moment_condition = boundary_info.get("moment", None)
        self._boundary_info = {"x": x_condition, "y": y_condition, "moment": moment_condition}
        self._rotation = rotation

    @property
    def rotation(self) -> float | None:
        return self._rotation

    @overload
    def boundary_condition(self, axis: _BoundaryInfoKeys) -> None | float: ...

    @overload
    def boundary_condition(
        self,
    ) -> _BoundaryCondition2DInfo: ...

    def boundary_condition(self, axis: _BoundaryInfoKeys | None = None) -> None | float | _BoundaryCondition2DInfo:
        if not axis:
            return {
                "x": self._boundary_info.get("x"),
                "y": self._boundary_info.get("y"),
                "moment": self._boundary_info.get("moment"),
            }

        return self._boundary_info.get(axis)

    @property
    def is_x_fixed(self) -> bool:
        return self._is_axis_fixed("x")

    @property
    def is_y_fixed(self) -> bool:
        return self._is_axis_fixed("y")

    @property
    def is_moment_fixed(self) -> bool:
        return self._is_axis_fixed("moment")

    @property
    def prescribed_x(self) -> tuple[Literal[True], float] | tuple[Literal[False], None]:
        return self._prescribed_value(self._boundary_info.get("x"))

    @property
    def prescribed_y(self) -> tuple[Literal[True], float] | tuple[Literal[False], None]:
        return self._prescribed_value(self._boundary_info.get("y"))

    @property
    def prescribed_moment(
        self,
    ) -> tuple[Literal[True], float] | tuple[Literal[False], None]:
        return self._prescribed_value(self._boundary_info.get("moment"))

    def _prescribed_value(self, value: None | float) -> tuple[Literal[True], float] | tuple[Literal[False], None]:
        if value is None:
            return False, None
        else:
            return True, value

    def _is_axis_fixed(self, axis: _BoundaryInfoKeys) -> bool:
        value = self._boundary_info.get(axis)

        if value is None or value != 0:
            return False
        else:
            return True

    def __str__(self) -> str:
        string = f"Condition[{repr(self._point)}]("

        if self._boundary_info["x"] is not None:
            string += f"x: {self._boundary_info['x']}"

        if self._boundary_info["y"] is not None:
            string += f", y: {self._boundary_info['y']}"

        if self._boundary_info["moment"] is not None:
            string += f", moment: {self._boundary_info['moment']}"

        if self._rotation is not None:
            string += f", rotation: {self._rotation}"

        return string + ")"
