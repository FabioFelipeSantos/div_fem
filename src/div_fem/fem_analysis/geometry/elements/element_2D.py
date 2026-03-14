from multiprocessing import Value
from typing import Literal, TypedDict, NotRequired, Callable

from div_fem.matrices.base_matrix import Matrix
from div_fem.fem_analysis.geometry.point import Point
from div_fem.matrices.base_vector import Vector
from div_fem.fem_analysis.loads.element_2D_loads import Element2DLoads
from div_fem.fem_analysis.shape_functions.shape_functions_2D_bar import (
    shape_functions_2D_bar,
    first_derivative_shape_functions_2D_bar,
)
from div_fem.algorithms.integration.gauss_quadrature import gauss_quadrature
from .base_element import BaseElement


class MaterialAndSectionGeometryProperties(TypedDict):
    E: float
    A: float
    I: float
    nu: NotRequired[float | None]


class Element2D(BaseElement):
    _type: Literal["bar", "beam", "frame"]
    _cosine_angles: tuple[float, float]
    _length: float
    _material_and_section_properties: MaterialAndSectionGeometryProperties
    _loads: list[Element2DLoads] | None

    def __init__(
        self,
        points: list[Point],
        degrees_of_freedom: list[int] | list[list[int]],
        material_and_section_properties: MaterialAndSectionGeometryProperties,
        type: Literal["bar", "beam", "frame"] = "bar",
        loads: list[Element2DLoads] | None = None,
    ) -> None:
        self._type = type
        super().__init__(points, degrees_of_freedom)
        self._length = self._calculating_length()
        self._cosine_angles = self._calculating_cosine_angles()
        if not material_and_section_properties.get("nu"):
            material_and_section_properties.update({"nu": None})

        self._material_and_section_properties = material_and_section_properties
        self._loads = loads

    @property
    def T(self) -> Matrix:
        """
        Rotation matrix for the 2D element to transform from local to global coordinates.
        """
        return Matrix(
            [
                [self._cosine_angles[0], self._cosine_angles[1], 0, 0],
                [0, 0, self._cosine_angles[0], self._cosine_angles[1]],
            ]
        )

    @property
    def local_stiffness_matrix(self) -> Matrix:
        return gauss_quadrature(
            _integration_bar_function_for_stiffness_matrix,
            1,
            E=self._material_and_section_properties["E"],
            A=self._material_and_section_properties["A"],
            L=self._length,
        )

    @property
    def local_forces_vector(self) -> Vector:
        if not self._loads:
            raise ValueError(
                "The element doesn't have any loads applied. Please create an Element2DLoads instance and provided in the definition of the element."
            )
        loads = self._loads
        equivalent_forces = self._base_vector_element()

        for load in loads:
            if load.force_type == "concentrated" or load.force_type == "moment":
                if not isinstance(load.force_value, (float, int)):
                    raise ValueError("Concentrated loads must be only a float value.")

                if not load.force_point:
                    raise ValueError(
                        "To concentrated forces a float value in [0, 1] must be provided as the point of the application of the force. x = 0 is the beginning of the element and x = 1 is the end of the element."
                    )

                equivalent_forces += _bar_concentrated_equivalent_forces_vector(
                    load.force_value, load.force_point
                )
            else:
                if load.force_type == "constant":
                    equivalent_forces += gauss_quadrature(
                        _integration_bar_function_for_forces_vector,
                        2,
                        force=load,
                        L=self._length,
                    )
                else:
                    equivalent_forces += gauss_quadrature(
                        _integration_bar_function_for_forces_vector,
                        6,
                        force=load,
                        L=self._length,
                    )

        return equivalent_forces

    @property
    def type(self) -> Literal["bar", "beam", "frame"]:
        return self._type

    @property
    def length(self) -> float:
        return self._length

    def _calculating_length(self) -> float:
        return (Vector(self._points[1]) - Vector(self._points[0])).norm

    def _calculating_cosine_angles(self) -> tuple[float, float]:
        c = (self._points[1][0] - self._points[0][0]) / self._length
        s = (self._points[1][1] - self._points[0][1]) / self._length

        return (c, s)

    def _base_matrix_element(self) -> Matrix:
        if self._type == "bar":
            return Matrix(rows=2)
        elif self._type == "beam":
            return Matrix(rows=4)
        else:
            return Matrix(rows=6)

    def _base_vector_element(self) -> Vector:
        if self._type == "bar":
            return Vector(rows=2)
        elif self._type == "beam":
            return Vector(rows=4)
        else:
            return Vector(rows=6)


def _integration_bar_function_for_stiffness_matrix(
    independent_value: float, **kwargs: float
) -> Matrix:
    E = kwargs.pop("E", None)
    A = kwargs.pop("A", None)
    L = kwargs.pop("L", None)

    if not E:
        raise ValueError(
            "The material parameter E (Youngs Modulus) must be passed to integration function."
        )
    if not A:
        raise ValueError(
            "The area for the cross section must be provided in the integration function."
        )
    if not L:
        raise ValueError(
            "The length of the bar must be passed to integration function."
        )

    B = first_derivative_shape_functions_2D_bar(independent_value) * (2 / L)

    return (B.T * B) * (E * A) * (L / 2)


def _bar_concentrated_equivalent_forces_vector(force: float, point: float) -> Vector:
    return Vector(shape_functions_2D_bar(2 * point - 1).get_list()[0]) * force


def _integration_bar_function_for_forces_vector(
    independent_value: float, **kwargs: float | Element2DLoads
) -> Vector:
    force = kwargs.pop("force", None)
    if not force:
        raise ValueError(
            "Some force must be provided to calculate the resulting vector of equivalent forces."
        )

    if not isinstance(force, Element2DLoads):
        raise ValueError("The forces must be a valid Element2DLoads class.")

    L = kwargs.pop("L", None)

    if not L:
        raise ValueError(
            "The length of the bar must be passed to integration function."
        )

    if not isinstance(L, float):
        raise ValueError("The length of the element must be a float")

    x_init = force.force_init_point if force.force_init_point else 0.0
    x_final = force.force_final_point if force.force_final_point else 1

    new_xi = -1 + (x_init + x_final) + (x_final - x_init) * independent_value

    N = Vector(shape_functions_2D_bar(new_xi).get_list()[0])

    if force.force_type == "constant":
        if not isinstance(force.force_value, (float | int)):
            raise ValueError(
                "For a constant distributed force, it magnitude must be a float number."
            )

        return N * force.force_value * (L / 2) * (x_final - x_init)
    else:
        # force.force_type == "function":
        if not isinstance(force.force_value, Callable):
            raise ValueError(
                "For a function force, provide a callable (float) -> float function."
            )
        return N * force.force_value((new_xi + 1) / 2) * (L / 2) * (x_final - x_init)
