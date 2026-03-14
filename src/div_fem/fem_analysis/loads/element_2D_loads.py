from typing import Literal, Callable, overload

_ForceTypes = Literal["concentrated", "moment", "constant", "function"]


class Element2DLoads:
    """
    The Element2DLoads is a class to set any kind of force that act in an 2D element of type bar, beam or frame. The possible types of loads is concentrated (perpendicular, axial or moment) and distributed (linear or by a function). As the element is in 2D the points where the forces act must be provided in the support coordinates interval [0, 1], where 0 represents the left end of the element and 1 represents the right end of the element. So, if a load is applied at 0.5, this represent the middle of the element.

    Some different types of definition is possible:

    To concentrated loads you must provided:
        type = "concentrated" | "moment"
        force_value (float): The magnitude of the force.
        force_point (float): The point in the support coordinates.

    To a constant distributed load you must provided:
        type = "constant"
        force_value (float): The magnitude of the constant distributed force.
        force_init_point (float): The initial point in support coordinates. If no value was set, the load begins at the left end of the element.
        force_final_point (float): The final point in support coordinates. If no value was set, the load ends at the right end of the element.

    To a load that isn't concentrated or constant, you can pass a distributed function as force, like w_0 * sin(pi * x), where you can define function = lambda x: w_0 * math.sin(math.pi * x). The function f(x) = P, where P is constant represents the previous case, so use type = "constant" in this case. Thus, to a function load you should set
        type = "function"
        force_value (Callable[[float], float]): The function that represents the load applied in the support coordinates.
        force_init_point (float): The initial point in support coordinates. If no value was set, the load begins at the left end of the element.
        force_final_point (float): The final point in support coordinates. If no value was set, the load ends at the right end of the element.
    """

    _type: _ForceTypes
    _force_value: float | Callable[[float], float] | None = None
    _force_point: float | None = None
    _force_init_point: float | None = None
    _force_final_point: float | None = None

    @overload
    def __init__(
        self,
        type: Literal["concentrated", "moment"],
        force_value: float,
        force_point: float,
    ) -> None: ...

    @overload
    def __init__(
        self,
        type: Literal["constant"],
        force_value: float,
        *,
        force_init_point: float | None = ...,
        force_final_point: float | None = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        type: Literal["function"],
        force_value: Callable[[float], float],
        *,
        force_init_point: float | None = ...,
        force_final_point: float | None = ...,
    ) -> None: ...

    def __init__(
        self,
        type: _ForceTypes,
        force_value: float | Callable[[float], float],
        force_point: float | None = None,
        *,
        force_init_point: float | None = None,
        force_final_point: float | None = None,
    ) -> None:
        self._type = type
        self._force_value = force_value
        self._force_point = force_point

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
    def force_value(self) -> float | Callable[[float], float] | None:
        return self._force_value

    @property
    def force_point(self) -> float | None:
        return self._force_point

    @property
    def force_init_point(self) -> float | None:
        return self._force_init_point

    @property
    def force_final_point(self) -> float | None:
        return self._force_final_point
