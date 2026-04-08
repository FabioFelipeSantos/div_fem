import math
from typing import Literal, overload

from div_fem.matrices.base_matrix import Matrix
from div_fem.fem_analysis.geometry.elements.shape_functions_interface import (
    ShapeFunctions,
)
from div_fem.matrices.base_vector import Vector

_TypeOf2DElement = Literal["bar", "beam", "frame"]

DERIVATIVE_ORDER_VALID = {1, 2, 3}


class ShapeFunctions2D(ShapeFunctions):
    _number_of_points: int
    _type: _TypeOf2DElement
    _total_degree_of_freedom: int

    def __init__(self, number_of_points: int, total_degree_of_freedom: int, length: float, type: _TypeOf2DElement = "bar") -> None:
        self._number_of_points = number_of_points
        self._total_degree_of_freedom = total_degree_of_freedom
        self._interpolation_points = self._calculating_interpolation_points()
        self._type = type
        self._barycentric_weights = self._barycentric_weights_calculation()

        if type != "bar":
            self._nodal_inclination = self._nodal_inclination_calculation()

        self._jacobian = length / 2

    @property
    def number_of_points(self) -> int:
        return self._number_of_points

    @property
    def interpolation_points(self) -> Vector:
        return self._interpolation_points

    @property
    def barycentric_weights(self) -> Vector:
        return self._barycentric_weights

    @property
    def nodal_inclination(self) -> Vector | None:
        return self._nodal_inclination

    @property
    def jacobian(self) -> float:
        return self._jacobian

    @overload
    def value(self, xi: float) -> Matrix: ...

    @overload
    def value(self, xi: float, index: int) -> float: ...

    @overload
    def value(self, xi: float, index: list[int]) -> list[float]: ...

    def value(self, xi: float, index: int | list[int] | None = None) -> Matrix | float | list[float]:
        if index:
            self._raise_error_for_index_greater_than_total_dof(index)

            if self._type == "bar":
                if not isinstance(index, list):
                    return self._Lagrangian_function(index, xi)
                else:
                    return [self._Lagrangian_function(idx, xi) for idx in index]
            elif self._type == "beam":
                if not isinstance(index, list):
                    nodal_number = index // 2
                    nodal_dof_zero_index = index % 2

                    return self._Hermite_function(nodal_number, xi)[nodal_dof_zero_index]
                else:
                    values: list[float] = []

                    for idx in index:
                        nodal_number = idx // 2
                        nodal_dof_zero_index = idx % 2

                        values.append(self._Hermite_function(nodal_number, xi)[nodal_dof_zero_index])

                    return values
            else:
                if not isinstance(index, list):
                    nodal_number = index // 3
                    nodal_dof_zero_index = index % 3

                    if nodal_dof_zero_index == 0:
                        return self._Lagrangian_function(nodal_number, xi)
                    else:
                        return self._Hermite_function(nodal_number, xi)[nodal_dof_zero_index - 1]
                else:
                    values = []

                    for idx in index:
                        nodal_number = idx // 3
                        nodal_dof_zero_index = idx % 3

                        if nodal_dof_zero_index == 0:
                            values.append(self._Lagrangian_function(nodal_number, xi))
                        else:
                            values.append(self._Hermite_function(nodal_number, xi)[nodal_dof_zero_index - 1])

                    return values

        if self._type == "bar":
            return Matrix([[self._Lagrangian_function(i, xi) for i in range(self.number_of_points)]])
        elif self._type == "beam":
            return self._Hermite_values(xi)
        else:
            return self._frame_shape_values(xi)

    @overload
    def derivative(
        self,
        diff_order: int,
        xi: float,
        *,
        is_for_stiffness_matrix: bool = False,
    ) -> Matrix: ...

    @overload
    def derivative(
        self,
        diff_order: int,
        xi: float,
        index: int,
        *,
        is_for_stiffness_matrix: bool = False,
    ) -> float: ...

    @overload
    def derivative(
        self,
        diff_order: int,
        xi: float,
        index: list[int],
        *,
        is_for_stiffness_matrix: bool = False,
    ) -> list[float]: ...

    def derivative(
        self,
        diff_order: int,
        xi: float,
        index: int | list[int] | None = None,
        *,
        is_for_stiffness_matrix: bool = False,
    ) -> Matrix | float | list[float]:
        if not diff_order in DERIVATIVE_ORDER_VALID:
            raise ValueError(
                f"The derivative of shape functions is only available for orders: {', '.join([str(value) for value in DERIVATIVE_ORDER_VALID])}"
            )

        if index:
            self._raise_error_for_index_greater_than_total_dof(index)

            if self._type == "bar":
                if not isinstance(index, list):
                    return self._Lagrangian_derivative(diff_order, index, xi)
                else:
                    return [self._Lagrangian_derivative(diff_order, idx, xi) for idx in index]
            elif self._type == "beam":
                if not isinstance(index, list):
                    return self._Hermite_derivative(diff_order, index, xi)
                else:
                    values: list[float] = []

                    for idx in index:
                        nodal_number = idx // 2
                        nodal_dof_zero_index = idx % 2

                        values.append(self._Hermite_derivative(diff_order, nodal_number, xi)[nodal_dof_zero_index])

                    return values
            else:
                if not isinstance(index, list):
                    nodal_number = index // 3
                    nodal_dof_zero_index = index % 3

                    if nodal_dof_zero_index == 0:
                        return self._Lagrangian_derivative(
                            (diff_order if not is_for_stiffness_matrix else diff_order - 1),
                            nodal_number,
                            xi,
                        )
                    else:
                        return self._Hermite_derivative(diff_order, nodal_number, xi)[nodal_dof_zero_index - 1]
                else:
                    values = []

                    for idx in index:
                        nodal_number = idx // 3
                        nodal_dof_zero_index = idx % 3

                        if nodal_dof_zero_index == 0:
                            values.append(
                                self._Lagrangian_derivative(
                                    (diff_order if not is_for_stiffness_matrix else diff_order - 1),
                                    nodal_number,
                                    xi,
                                )
                            )
                        else:
                            values.append(self._Hermite_derivative(diff_order, nodal_number, xi)[nodal_dof_zero_index - 1])

                    return values
        else:
            if self._type == "bar":
                return Matrix([[self._Lagrangian_derivative(diff_order, i, xi) for i in range(self.number_of_points)]])
            elif self._type == "beam":
                return self._Hermite_derivative_values(diff_order, xi)
            else:
                return self._frame_shape_derivative_values(diff_order, xi, is_for_stiffness_matrix=is_for_stiffness_matrix)

    def _calculating_interpolation_points(self) -> Vector:
        h = 2 / (self._number_of_points - 1)

        return Vector([-1 + t * h for t in range(0, self._number_of_points)])

    def _barycentric_weights_calculation(self) -> Vector:
        barycentric_weights = Vector(rows=self.number_of_points)

        for i, x_i in self.interpolation_points.enumeration:
            prod = 1.0

            for j, x_j in self._interpolation_points.enumeration:
                if j != i:
                    prod *= 1 / (x_i - x_j)

            barycentric_weights[i] = prod

        return barycentric_weights

    def _nodal_poly(self, xi) -> float:
        prod = 1.0

        for k, x_k in self.interpolation_points.enumeration:
            prod *= xi - x_k

        return prod

    def _Lagrangian_function(self, i: int, xi: float) -> float:
        if xi in self.interpolation_points:
            if xi == self.interpolation_points[i]:
                return 1.0
            else:
                return 0.0
        else:
            return self._nodal_poly(xi) * (self.barycentric_weights[i] / (xi - self.interpolation_points[i]))

    def _logarithmic_sums(self, sum_order: int, i: int, xi) -> float:
        sum = 0

        for k, x_k in self.interpolation_points.enumeration:
            if k != i:
                sum += 1 / (math.pow(xi - x_k, sum_order))

        return sum

    def _Lagrangian_derivative(self, diff_order: int, i: int, xi: float) -> float:

        if xi not in self.interpolation_points or xi == self.interpolation_points[i]:
            Lagrangian_function_value = self._Lagrangian_function(i, xi)

            if diff_order == 1:
                return Lagrangian_function_value * self._logarithmic_sums(1, i, xi)
            elif diff_order == 2:
                S1 = self._logarithmic_sums(1, i, xi)
                S2 = self._logarithmic_sums(2, i, xi)

                return Lagrangian_function_value * (S1**2 - S2)
            else:
                S1 = self._logarithmic_sums(1, i, xi)
                S2 = self._logarithmic_sums(2, i, xi)
                S3 = self._logarithmic_sums(3, i, xi)

                return Lagrangian_function_value * (S1**3 - 3 * S1 * S2 + 2 * S3)
        else:
            index = self.interpolation_points.index(xi)
            x_j = xi
            x_i = self.interpolation_points[i]
            barycentric_ratio = self.barycentric_weights[i] / self.barycentric_weights[index]

            if diff_order == 1:
                return barycentric_ratio * (1 / (x_j - x_i))
            elif diff_order == 2:
                return (2 / (x_j - x_i)) * (
                    barycentric_ratio * self._Lagrangian_derivative(diff_order - 1, index, xi) - self._Lagrangian_derivative(diff_order - 1, i, x_j)
                )
            else:
                return (3 / (x_j - x_i)) * (
                    barycentric_ratio * self._Lagrangian_derivative(diff_order - 1, index, xi) - self._Lagrangian_derivative(diff_order - 1, i, x_j)
                )

    def _nodal_inclination_calculation(self) -> Vector:
        nodal_inclination = Vector(rows=self.number_of_points)

        for i, x_i in self.interpolation_points.enumeration:
            nodal_inclination[i] = self._Lagrangian_derivative(1, i, x_i)

        return nodal_inclination

    def _Hermite_function(self, i: int, xi: float) -> list[float]:
        if not self.nodal_inclination:
            raise ValueError("To calculate Hermite functions, the nodal inclination must be calculated before.")

        w = xi - self.interpolation_points[i]
        u = 1 - 2 * self.nodal_inclination[i] * w
        v = self._Lagrangian_function(i, xi) ** 2

        return [u * v, (w * v) * self.jacobian]

    def _Hermite_derivative(self, diff_order: int, i: int, xi: float) -> list[float]:
        if not self.nodal_inclination:
            raise ValueError("To calculate Hermite functions, the nodal inclination must be calculated before.")

        u_derivative = -2 * self.nodal_inclination[i]
        w = xi - self.interpolation_points[i]
        u = 1 - 2 * self.nodal_inclination[i] * w

        if diff_order == 1:
            v = self._Lagrangian_function(i, xi) ** 2
            v_first_derivate = 2 * self._Lagrangian_function(i, xi) * self._Lagrangian_derivative(1, i, xi)

            return [
                u_derivative * v + u * v_first_derivate,
                (v + w * v_first_derivate) * self.jacobian,
            ]
        elif diff_order == 2:
            v_first_derivate = 2 * self._Lagrangian_function(i, xi) * self._Lagrangian_derivative(1, i, xi)
            v_second_derivative = 2 * self._Lagrangian_derivative(1, i, xi) ** 2 + 2 * self._Lagrangian_function(i, xi) * self._Lagrangian_derivative(
                2, i, xi
            )

            return [
                2 * u_derivative * v_first_derivate + u * v_second_derivative,
                (2 * v_first_derivate + w * v_second_derivative) * self.jacobian,
            ]
        else:
            v_second_derivative = 2 * self._Lagrangian_derivative(1, i, xi) ** 2 + 2 * self._Lagrangian_function(i, xi) * self._Lagrangian_derivative(
                2, i, xi
            )
            v_third_derivative = 6 * self._Lagrangian_derivative(1, i, xi) * self._Lagrangian_derivative(2, i, xi) + 2 * self._Lagrangian_function(
                i, xi
            ) * self._Lagrangian_derivative(3, i, xi)

            return [
                3 * u_derivative * v_second_derivative + u * v_third_derivative,
                (3 * v_second_derivative + w * v_third_derivative) * self.jacobian,
            ]

    def _Hermite_values(self, xi: float) -> Matrix:
        Hermite_values = Matrix(rows=1, columns=self.number_of_points * 2)

        for i in range(self.number_of_points):
            Hermite_values[0, 2 * i], Hermite_values[0, 2 * i + 1] = self._Hermite_function(i, xi)

        return Hermite_values

    def _Hermite_derivative_values(self, diff_order: int, xi: float) -> Matrix:
        Hermite_derivative_values = Matrix(rows=1, columns=self.number_of_points * 2)

        for i in range(self.number_of_points):
            (
                Hermite_derivative_values[0, 2 * i],
                Hermite_derivative_values[0, 2 * i + 1],
            ) = self._Hermite_derivative(diff_order, i, xi)

        return Hermite_derivative_values

    def _frame_shape_values(self, xi: float) -> Matrix:
        frame_values = Matrix(rows=1, columns=self.number_of_points * 3)

        for i in range(self.number_of_points):
            frame_values[0, 3 * i] = self._Lagrangian_function(i, xi)
            frame_values[0, 3 * i + 1], frame_values[0, 3 * i + 2] = self._Hermite_function(i, xi)

        return frame_values

    def _frame_shape_derivative_values(self, diff_order: int, xi: float, is_for_stiffness_matrix: bool) -> Matrix:
        frame_shape_derivative_values = Matrix(rows=1, columns=self.number_of_points * 3)

        for i in range(self.number_of_points):
            frame_shape_derivative_values[0, 3 * i] = self._Lagrangian_derivative(
                diff_order if not is_for_stiffness_matrix else diff_order - 1, i, xi
            )
            (
                frame_shape_derivative_values[0, 3 * i + 1],
                frame_shape_derivative_values[0, 3 * i + 2],
            ) = self._Hermite_derivative(diff_order, i, xi)

        return frame_shape_derivative_values

    def _raise_error_for_index_greater_than_total_dof(self, index: int | list[int]) -> None:
        if not isinstance(index, list):
            if index > self._total_degree_of_freedom - 1:
                raise IndexError(
                    f"The shape function must have a index in 0 indexation and by DOF of node. Received {index}, expected at maximum of {self._total_degree_of_freedom-1}."
                )
        else:
            for idx in index:
                if idx > self._total_degree_of_freedom - 1:
                    raise IndexError(
                        f"The shape function must have a index in 0 indexation and by DOF of node. Received {idx}, expected at maximum of {self._total_degree_of_freedom-1}."
                    )
