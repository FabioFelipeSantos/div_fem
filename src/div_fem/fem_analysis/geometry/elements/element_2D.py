from typing import Literal, TypedDict, NotRequired, Callable

from div_fem.matrices.base_matrix import Matrix
from div_fem.fem_analysis.geometry.point import Point
from div_fem.matrices.base_vector import Vector
from div_fem.fem_analysis.loads.element_2D_loads import Element2DLoads
from div_fem.fem_analysis.shape_functions.shape_functions_2D_bar import (
    ShapeFunctions2D,
    _TypeOf2DElement,
)
from div_fem.algorithms.integration.gauss_quadrature import gauss_quadrature
from .base_element import BaseElement


class MaterialAndSectionGeometryProperties(TypedDict):
    E: float
    A: float
    I: float
    nu: NotRequired[float | None]


class Element2D(BaseElement):
    _type: _TypeOf2DElement
    _cosine_angles: tuple[float, float]
    _length: float
    _material_and_section_properties: MaterialAndSectionGeometryProperties
    _loads: list[Element2DLoads] | None

    def __init__(
        self,
        points: list[Point],
        number_of_points_of_interpolation: int,
        degrees_of_freedom: list[int] | list[list[int]],
        material_and_section_properties: MaterialAndSectionGeometryProperties,
        type: _TypeOf2DElement = "bar",
        loads: list[Element2DLoads] | None = None,
    ) -> None:
        self._type = type
        self._total_element_degree_of_freedom = self._verifying_dof_number(
            number_of_points_of_interpolation, degrees_of_freedom
        )
        super().__init__(points, number_of_points_of_interpolation, degrees_of_freedom)
        self._length = self._calculating_length()
        self._cosine_angles = self._calculating_cosine_angles()
        if not material_and_section_properties.get("nu"):
            material_and_section_properties.update({"nu": None})

        self._material_and_section_properties = material_and_section_properties
        self._loads = loads
        self._shape_functions = ShapeFunctions2D(
            self.number_points,
            self._total_element_degree_of_freedom,
            self.length,
            self.type,
        )

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
        if self.type == "bar":
            return gauss_quadrature(
                self._integration_function_for_stiffness_matrix,
                1,
                E=self._material_and_section_properties["E"],
                A=self._material_and_section_properties["A"],
                L=self._length,
            )
        elif self.type == "beam":
            return gauss_quadrature(
                self._integration_function_for_stiffness_matrix,
                self.number_points,
                E=self._material_and_section_properties["E"],
                I=self._material_and_section_properties["I"],
                L=self._length,
            )
        else:
            return gauss_quadrature(
                self._integration_function_for_stiffness_matrix,
                self.number_points,
                E=self._material_and_section_properties["E"],
                A=self._material_and_section_properties["A"],
                I=self._material_and_section_properties["I"],
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
                if not load.force_point:
                    raise ValueError(
                        "To concentrated forces a float value in [0, 1] must be provided as the point of the application of the force. x = 0 is the beginning of the element and x = 1 is the end of the element."
                    )

                equivalent_forces += self._concentrated_equivalent_forces_vector(
                    load, load.force_point
                )
            else:
                if load.force_type == "constant":
                    equivalent_forces += gauss_quadrature(
                        self._integration_function_for_forces_vector,
                        2,
                        load=load,
                        L=self._length,
                    )
                else:
                    equivalent_forces += gauss_quadrature(
                        self._integration_function_for_forces_vector,
                        6,
                        load=load,
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
        return Matrix(rows=self.element_degree_of_freedom)

    def _base_vector_element(self) -> Vector:
        return Vector(rows=self.element_degree_of_freedom)

    def _integration_function_for_stiffness_matrix(
        self, independent_value: float, **kwargs: float
    ) -> Matrix:
        E = kwargs.pop("E", None)
        A = kwargs.pop("A", None)
        I = kwargs.pop("I", None)
        L = kwargs.pop("L", None)

        if not E:
            raise ValueError(
                "The material parameter E (Youngs Modulus) must be passed to integration function."
            )

        if not L:
            raise ValueError(
                "The length of the bar must be passed to integration function."
            )

        if self.type == "bar" or self.type == "frame":
            if not A:
                raise ValueError(
                    "The area for the cross section must be provided in the integration function."
                )

        if self.type == "beam" or self.type == "frame":
            if not I:
                raise ValueError(
                    "The moment of inertia for the cross section must be provided in the integration function."
                )

        if self.type == "bar" and A:
            B = self.shape_functions.derivative(1, independent_value) * (
                1 / self.shape_functions.jacobian
            )

            return (B.T * B) * (E * A) * self.shape_functions.jacobian

        if self.type == "beam" and I:
            B = (
                self.shape_functions.derivative(2, independent_value)
                * (1 / self.shape_functions.jacobian) ** 2
            )

            return (B.T * B) * (E * I) * self.shape_functions.jacobian

        if self.type == "frame" and A and I:
            bar_indices = [3 * i for i in range(self.number_points)]
            beam_indices = [
                idx for i in range(self.number_points) for idx in [3 * i + 1, 3 * i + 2]
            ]

            B_aux = self.shape_functions.derivative(
                2, independent_value, is_for_stiffness_matrix=True
            )
            B_bar = B_aux[[0], bar_indices] * (1 / self.shape_functions.jacobian)
            B_beam = B_aux[[0], beam_indices] * (1 / self.shape_functions.jacobian) ** 2

            frame_matrix = Matrix(rows=3 * self.number_points)
            frame_matrix[bar_indices] = (B_bar.T * B_bar) * (E * A)
            frame_matrix[beam_indices] = (B_beam.T * B_beam) * (E * I)

            return frame_matrix * self.shape_functions.jacobian

        raise ValueError(
            "Some problem in the definition of your functions to integrate."
        )

    def _concentrated_equivalent_forces_vector(
        self, load: Element2DLoads, point: float
    ) -> Vector:
        point = 2 * point - 1

        if self.type == "bar":
            force_x = load.force_value("x")
            if not isinstance(force_x, (float | int)):
                raise ValueError(
                    "The value for a concentrated force or moment must be a valid float number."
                )

            return self.shape_functions.value(point).to_vector() * force_x
        elif self.type == "beam":
            force_y, moment = load.force_value(["y", "moment"])

            if not force_y and not moment:
                raise ValueError(
                    "To concentrated loads in beams, provide a value for y axis or for moment."
                )
            elif load.force_type == "moment" and not moment:
                raise ValueError(
                    "To concentrated moment in beams, provide a float value for the moment."
                )
            else:
                force_vector = Vector(rows=self.element_degree_of_freedom)

                if force_y:

                    if not isinstance(force_y, (float | int)):
                        raise ValueError(
                            "To concentrated y axis load provide a float value."
                        )
                    force_vector += (
                        self.shape_functions.value(point).to_vector() * force_y
                    )

                if moment:
                    if not isinstance(moment, (float | int)):
                        raise ValueError(
                            "To concentrated y axis load provide a float value."
                        )

                    force_vector += (
                        self.shape_functions.derivative(1, point).to_vector()
                        * moment
                        * (1 / self.shape_functions.jacobian)
                    )

                return force_vector
        else:
            force_x, force_y, moment = load.force_value(["x", "y", "moment"])

            if not force_x and not force_y and not moment:
                raise ValueError(
                    "To concentrated loads in frames, provide a value for x or y axis or for moment."
                )
            elif load.force_type == "moment" and not moment:
                raise ValueError(
                    "To concentrated moment in frames, provide a float value for the moment."
                )
            else:
                force_vector = Vector(rows=self.element_degree_of_freedom)

                if force_x or force_y:
                    N = self.shape_functions.value(point).to_vector()

                    if force_x:
                        if not isinstance(force_x, (float | int)):
                            raise ValueError(
                                "To concentrated x axis load provide a float value."
                            )

                        force_vector[[0, 3]] += N[[0, 3]] * force_x

                    if force_y:
                        if not isinstance(force_y, (float | int)):
                            raise ValueError(
                                "To concentrated x axis load provide a float value."
                            )

                        force_vector[[1, 2, 4, 5]] = N[[1, 2, 4, 5]] * force_y

                if moment:
                    if not isinstance(moment, (float | int)):
                        raise ValueError(
                            "To concentrated y axis load provide a float value."
                        )

                    dN = self.shape_functions.derivative(1, point).to_vector()

                    force_vector[[1, 2, 4, 5]] += (
                        dN[[1, 2, 4, 5]] * moment * (1 / self.shape_functions.jacobian)
                    )

                return force_vector

    def _integration_function_for_forces_vector(
        self, independent_value: float, **kwargs: float | Element2DLoads
    ) -> Vector:
        load = kwargs.pop("load", None)

        if not load:
            raise ValueError(
                "Some force must be provided to calculate the resulting vector of equivalent forces."
            )

        if not isinstance(load, Element2DLoads):
            raise ValueError("The forces must be a valid Element2DLoads class.")

        L = kwargs.pop("L", None)

        if not L:
            raise ValueError(
                "The length of the bar must be passed to integration function."
            )

        if not isinstance(L, float):
            raise ValueError("The length of the element must be a float")

        x_init = load.force_init_point if load.force_init_point else 0.0
        x_final = load.force_final_point if load.force_final_point else 1.0

        new_xi = -1 + (x_init + x_final) + (x_final - x_init) * independent_value

        N = self.shape_functions.value(new_xi).to_vector()

        if self.type == "bar":
            force_x = load.force_value("x")

            if not force_x:
                raise ValueError(
                    "To apply loads on an bar element, provide a float or Callable[[float], float] to the atribute force_value_x for the load."
                )

            if load.force_type == "constant":
                if not isinstance(force_x, (float | int)):
                    raise ValueError(
                        "Provide a valid float for constant distributed load value for the x axis for elements of type bar."
                    )

                return N * force_x * (L / 2) * (x_final - x_init)
            else:
                if not isinstance(force_x, Callable):
                    raise ValueError(
                        "For a function force in a bar, provide a callable (float) -> float function."
                    )

                return N * force_x((new_xi + 1) / 2) * (L / 2) * (x_final - x_init)
        elif self.type == "beam":
            force_y, moment = load.force_value(["y", "moment"])

            if not force_y and not moment:
                raise ValueError(
                    "To apply loads on an bar element, provide a float or Callable[[float], float] to the force_value_y or force_value_moment atributes for the load."
                )

            force_value_vector = Vector(rows=self.element_degree_of_freedom)

            if load.force_type == "constant":
                if force_y:
                    if not isinstance(force_y, (float | int)):
                        raise ValueError(
                            "Provide a valid float for constant load values for the y axis for elements of type beam."
                        )

                    force_value_vector += N * force_y * (L / 2) * (x_final - x_init)

                if moment:
                    if not isinstance(moment, (float | int)):
                        raise ValueError(
                            "Provide a valid float for constant load values for moment for elements of type beam."
                        )

                    dN = self.shape_functions.derivative(1, new_xi).to_vector()

                    force_value_vector += (
                        (1 / self.shape_functions.jacobian)
                        * dN
                        * moment
                        * (L / 2)
                        * (x_final - x_init)
                    )
            else:
                if force_y:
                    if not isinstance(force_y, Callable):
                        raise ValueError(
                            "For a function force in y axis, provide a callable (float) -> float function to force_value_y."
                        )

                    force_value_vector += (
                        N * force_y((new_xi + 1) / 2) * (L / 2) * (x_final - x_init)
                    )

                if moment:
                    if not isinstance(moment, Callable):
                        raise ValueError(
                            "For a function force to moment, provide a callable (float) -> float function to force_value_moment."
                        )

                    dN = self.shape_functions.derivative(1, new_xi).to_vector()

                    force_value_vector += (
                        (1 / self.shape_functions.jacobian)
                        * dN
                        * moment((new_xi + 1) / 2)
                        * (L / 2)
                        * (x_final - x_init)
                    )

            return force_value_vector
        else:
            force_x, force_y, moment = load.force_value(["x", "y", "moment"])

            if not force_x and not force_y and not moment:
                raise ValueError(
                    "To apply loads on an frame element, provide a float or Callable[[float], float] to the force_value_x, force_value_y or force_value_moment atributes for the load."
                )

            force_value_vector = Vector(rows=self.element_degree_of_freedom)

            if load.force_type == "constant":
                if force_x:
                    if not isinstance(force_x, (float | int)):
                        raise ValueError(
                            "Provide a valid float for constant load values for the x axis for elements of type frame."
                        )

                    force_value_vector[[0, 3]] += (
                        N[[0, 3]] * force_x * (L / 2) * (x_final - x_init)
                    )

                if force_y:
                    if not isinstance(force_y, (float | int)):
                        raise ValueError(
                            "Provide a valid float for constant load values for the y axis for elements of type frame."
                        )

                    force_value_vector[[1, 2, 4, 5]] += (
                        N[[1, 2, 4, 5]] * force_y * (L / 2) * (x_final - x_init)
                    )

                if moment:
                    if not isinstance(moment, (float | int)):
                        raise ValueError(
                            "Provide a valid float for constant load values for moment for elements of type frame."
                        )

                    dN = self.shape_functions.derivative(1, new_xi).to_vector()

                    force_value_vector[[1, 2, 4, 5]] += (
                        (1 / self.shape_functions.jacobian)
                        * dN[[1, 2, 4, 5]]
                        * moment
                        * (L / 2)
                        * (x_final - x_init)
                    )
            else:
                if force_x:
                    if not isinstance(force_x, Callable):
                        raise ValueError(
                            "Provide a valid Callable[[float], float] for function distributed load values for the x axis for elements of type frame."
                        )

                    force_value_vector[[0, 3]] += (
                        N[[0, 3]]
                        * force_x((new_xi + 1) / 2)
                        * (L / 2)
                        * (x_final - x_init)
                    )

                if force_y:
                    if not isinstance(force_y, Callable):
                        raise ValueError(
                            "Provide a valid Callable[[float], float] for function distributed load values for the y axis for elements of type frame."
                        )

                    force_value_vector[[1, 2, 4, 5]] += (
                        N[[1, 2, 4, 5]]
                        * force_y((new_xi + 1) / 2)
                        * (L / 2)
                        * (x_final - x_init)
                    )

                if moment:
                    if not isinstance(moment, Callable):
                        raise ValueError(
                            "Provide a valid Callable[[float], float] for function distributed load values for the moment for elements of type frame."
                        )

                    dN = self.shape_functions.derivative(1, new_xi).to_vector()

                    force_value_vector[[1, 2, 4, 5]] += (
                        (1 / self.shape_functions.jacobian)
                        * dN[[1, 2, 4, 5]]
                        * moment((new_xi + 1) / 2)
                        * (L / 2)
                        * (x_final - x_init)
                    )

            return force_value_vector

    def _verifying_dof_number(
        self, number_of_points: int, degrees_of_freedom: list[int] | list[list[int]]
    ) -> int:
        if len(degrees_of_freedom) != number_of_points:
            raise ValueError(
                f"Provide just one information for degree of freedom for each node. Received {number_of_points} nodes and {len(degrees_of_freedom)} DOF info."
            )

        if isinstance(degrees_of_freedom[0], list):
            for dof_item in degrees_of_freedom:
                if not isinstance(dof_item, list):
                    raise TypeError(
                        "The degree of freedom list can't contain a mixture of ints and lists."
                    )

        if self._type == "bar":
            if not isinstance(degrees_of_freedom[0], int):
                raise ValueError(
                    "To an element of type bar, each point must have just one degree of freedom."
                )
            else:
                total = number_of_points * 1
                total_dof = len(degrees_of_freedom)

                if total != total_dof:
                    raise ValueError(
                        f"The total number of degree of freedom for the element must be {total} but received {total_dof}."
                    )
                return total_dof
        else:
            if not isinstance(degrees_of_freedom[0], list):
                raise ValueError(
                    "To an element of type beam or frame, each point must have a list of degree of freedom, one degree of each axis of deformation."
                )
            else:
                total_dof = 0

                if self._type == "beam":
                    total = number_of_points * 2

                    for index, dof in enumerate(degrees_of_freedom):
                        if isinstance(dof, list):
                            if len(dof) != 2:
                                raise ValueError(
                                    f"For a beam element, each node must have 2 degree of freedom. Received {len(dof)} for the node number {index + 1}"
                                )
                            total_dof += 2
                else:
                    total = number_of_points * 3

                    for index, dof in enumerate(degrees_of_freedom):
                        if isinstance(dof, list):
                            if len(dof) != 3:
                                raise ValueError(
                                    f"For a beam element, each node must have 2 degree of freedom. Received {len(dof)} for the node number {index + 1}"
                                )
                            total_dof += 3

                if total_dof != total:
                    raise ValueError(
                        f"The total number of degree of freedom for the element must be {total} but received {total_dof}."
                    )

                return total_dof
