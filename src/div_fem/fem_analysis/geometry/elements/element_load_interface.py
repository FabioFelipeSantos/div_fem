from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar, Generic, Mapping, Union, overload


if TYPE_CHECKING:
    from div_fem.fem_analysis.geometry.point import Point


_ForceType = TypeVar("_ForceType", bound=str)
_ForceAxis = TypeVar("_ForceAxis", bound=str)
_ForceValueType = TypeVar("_ForceValueType")
_ForceValue = TypeVar("_ForceValue", bound=Union[Mapping[str, Any], float])
_ForcePoint = TypeVar("_ForcePoint", bound=Union[float, "Point", None])


class ElementLoadInterface(
    ABC, Generic[_ForceType, _ForceAxis, _ForceValueType, _ForceValue, _ForcePoint]
):
    _type: _ForceType
    _force_value: _ForceValue
    _force_point: _ForcePoint

    @property
    @abstractmethod
    def force_type(self) -> _ForceType: ...

    @property
    @abstractmethod
    def force_point(self) -> _ForcePoint: ...

    @overload
    @abstractmethod
    def force_value(self) -> _ForceValue: ...

    @overload
    @abstractmethod
    def force_value(self, force_axis: _ForceAxis) -> _ForceValueType: ...

    @overload
    @abstractmethod
    def force_value(self, force_axis: list[_ForceAxis]) -> list[_ForceValueType]: ...
