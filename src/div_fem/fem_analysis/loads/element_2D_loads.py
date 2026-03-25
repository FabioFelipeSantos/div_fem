from ast import Call
import inspect
from typing import Literal, Callable, TypedDict, cast, overload, NotRequired

_ForceTypes = Literal["concentrated", "moment", "constant", "function"]


class ForceValueDict(TypedDict):
    x: NotRequired[float | Callable[[float], float] | None]
    y: NotRequired[float | Callable[[float], float] | None]
    moment: NotRequired[float | Callable[[float], float] | None]


class Element2DLoads:
    """
    The Element2DLoads is a class to set any kind of force that act in an 2D element of type bar, beam or frame. The possible types of loads is concentrated (perpendicular, axial or moment) and distributed (linear or by a function). As the element is in 2D the points where the forces act must be provided in the support coordinates interval [0, 1], where 0 represents the left end of the element and 1 represents the right end of the element. So, if a load is applied at 0.5, this represent the middle of the element.

    To define a value for the force, three different forces can be provided:
        force_value_x: float | Callable[[float], float] | None = None,
        force_value_y: float | Callable[[float], float] | None = None,
        force_value_moment: float | Callable[[float], float] | None = None,

    If the force isn't a function, pass just a float value.

    If none of these tree are defined, a ValueError will be raised.

    Beside the keyword arg for the value, some different types of definition will be necessary to define a element load. Follow the next rules:

    To concentrated loads you must provided:
        type = "concentrated" | "moment"
        force_point (float): The point in the support coordinates.

    To a constant distributed load you must provided:
        type = "constant"
        force_init_point (float): The initial point in support coordinates. If no value was set, the load begins at the left end of the element.
        force_final_point (float): The final point in support coordinates. If no value was set, the load ends at the right end of the element.

    To a load that isn't concentrated or constant, you can pass a distributed function as force, like w_0 * sin(pi * x), where you can define function = lambda x: w_0 * math.sin(math.pi * x). The function f(x) = P, where P is constant represents the previous case, so use type = "constant" in this case. Thus, to a function load you should set
        type = "function"
        force_init_point (float): The initial point in support coordinates. If no value was set, the load begins at the left end of the element.
        force_final_point (float): The final point in support coordinates. If no value was set, the load ends at the right end of the element.
    """

    _type: _ForceTypes
    _force_value_dict: ForceValueDict
    _force_point: float | None = None
    _force_init_point: float | None = None
    _force_final_point: float | None = None

    @overload
    def __init__(
        self,
        type: Literal["concentrated", "moment"],
        *,
        force_value_x: float | Callable[[float], float] | None = None,
        force_value_y: float | Callable[[float], float] | None = None,
        force_value_moment: float | Callable[[float], float] | None = None,
        force_point: float,
    ) -> None: ...

    @overload
    def __init__(
        self,
        type: Literal["constant"],
        *,
        force_value_x: float | Callable[[float], float] | None = None,
        force_value_y: float | Callable[[float], float] | None = None,
        force_value_moment: float | Callable[[float], float] | None = None,
        force_init_point: float | None = ...,
        force_final_point: float | None = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        type: Literal["function"],
        *,
        force_value_x: Callable[[float], float] | None = None,
        force_value_y: Callable[[float], float] | None = None,
        force_value_moment: Callable[[float], float] | None = None,
        force_init_point: float | None = ...,
        force_final_point: float | None = ...,
    ) -> None: ...

    def __init__(
        self,
        type: _ForceTypes,
        force_point: float | None = None,
        *,
        force_value_x: float | Callable[[float], float] | None = None,
        force_value_y: float | Callable[[float], float] | None = None,
        force_value_moment: float | Callable[[float], float] | None = None,
        force_init_point: float | None = None,
        force_final_point: float | None = None,
    ) -> None:
        if not force_value_x and not force_value_y and not force_value_moment:
            raise ValueError(
                "To create an element load provide at least one type of value for the load."
            )

        self._type = type
        self._force_point = force_point
        self._force_value_dict = {
            "x": force_value_x,
            "y": force_value_y,
            "moment": force_value_moment,
        }

        if (
            self._type == "concentrated" or self._type == "moment"
        ) and not self._force_point:
            raise ValueError(
                "A concentrated or moment load must be provided with a local x coordinate in the element. x in [0, 1], when x = 0 is the begging of the bar and x = 1 is the final of the bar."
            )

        if (
            not (self._type == "concentrated" or self._type == "moment")
            and self._force_point
        ):
            raise ValueError(
                "A non concentrated force cannot be provided with a point of application."
            )

        self._force_point = force_point
        self._force_init_point = force_init_point
        self._force_final_point = force_final_point

    @property
    def force_type(self) -> _ForceTypes:
        return self._type

    @property
    def force_point(self) -> float | None:
        return self._force_point

    @property
    def force_init_point(self) -> float | None:
        return self._force_init_point

    @property
    def force_final_point(self) -> float | None:
        return self._force_final_point

    @overload
    def force_value(self) -> tuple[
        float | Callable[[float], float] | None,
        float | Callable[[float], float] | None,
        float | Callable[[float], float] | None,
    ]: ...

    @overload
    def force_value(
        self, force_axis: Literal["x", "y", "moment"]
    ) -> float | Callable[[float], float] | None: ...

    @overload
    def force_value(
        self, force_axis: list[Literal["x", "y", "moment"]]
    ) -> list[float | Callable[[float], float] | None]: ...

    def force_value(
        self,
        force_axis: (
            Literal["x", "y", "moment"] | list[Literal["x", "y", "moment"]] | None
        ) = None,
    ) -> (
        float
        | Callable[[float], float]
        | None
        | list[float | Callable[[float], float] | None]
        | tuple[
            float | Callable[[float], float] | None,
            float | Callable[[float], float] | None,
            float | Callable[[float], float] | None,
        ]
    ):
        if not force_axis:
            return (
                self._force_value_dict.get("x"),
                self._force_value_dict.get("y"),
                self._force_value_dict.get("moment"),
            )

        if force_axis:
            if not isinstance(force_axis, list):
                if force_axis == "x":
                    return self._force_value_dict.get("x")
                elif force_axis == "y":
                    return self._force_value_dict.get("y")
                else:
                    return self._force_value_dict.get("moment")
            else:
                return [
                    cast(
                        float | Callable[[float], float] | None,
                        self._force_value_dict.get(key),
                    )
                    for key in force_axis
                ]

    def __str__(self) -> str:
        begin = f"Force({self._type}, value = "
        value = self.force_value()

        if self._type == "function":
            aux: list[str] = []
            for val in value:
                if val and callable(val):
                    try:
                        source = inspect.getsource(val).strip()
                        if "lambda" in source:
                            start = source.find("lambda")
                            source = source[start:].split(",")[0].split(")")[0]
                            aux.append(source)
                        elif "def" in source:
                            start = source.find("def") + 4
                            source = source[start:].split("\n")[0][:-1]
                        else:
                            source = "function"

                        aux.append(source)
                    except Exception:
                        aux.append("function")
                else:
                    aux.append("None")
            begin += f"({", ".join(aux)})"
        else:
            begin += str(value)

        middle = f", point = {self._force_point}" if self._force_point else ""
        if self._type == "constant" or self._type == "function":
            middle += f", init_point = {self._force_init_point if self._force_init_point else 0}"
            middle += f", final_point = {self._force_final_point if self._force_final_point else 1}"
        end = ")"

        return begin + middle + end
