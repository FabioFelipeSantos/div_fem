from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Mapping, Sequence, TypeVar

from div_fem.fem_analysis.geometry.elements.element_load_interface import (
    ElementLoadInterface,
)
from div_fem.fem_analysis.geometry.elements.shape_functions_interface import (
    ShapeFunctions,
)
from div_fem.fem_analysis.geometry.point import Point
from div_fem.matrices.base_matrix import Matrix
from div_fem.matrices.base_vector import Vector

if TYPE_CHECKING:
    from div_fem.fem_analysis.geometry.elements_container import Elements

_GeometryProperties = TypeVar(
    "_GeometryProperties", bound=Mapping[str, Any], covariant=True
)
_MaterialAndSectionProperties = TypeVar(
    "_MaterialAndSectionProperties", bound=Mapping[str, Any], covariant=True
)
_ElementType = TypeVar("_ElementType", bound=str, covariant=True)
_ShapeFunctions = TypeVar("_ShapeFunctions", bound=ShapeFunctions, covariant=True)
_ElementLoad = TypeVar("_ElementLoad", bound=ElementLoadInterface, covariant=True)


class ElementInterface(
    ABC,
    Generic[
        _ElementType,
        _GeometryProperties,
        _MaterialAndSectionProperties,
        _ShapeFunctions,
        _ElementLoad,
    ],
):
    _type: _ElementType
    _points: list[Point]
    _shape_functions: _ShapeFunctions
    _geometry_properties: _GeometryProperties
    _material_and_section_properties: _MaterialAndSectionProperties
    _loads: list[_ElementLoad] | None

    @property
    @abstractmethod
    def shape_functions(self) -> ShapeFunctions: ...

    @property
    @abstractmethod
    def T(self) -> Matrix:
        """
        Rotation matrix for the 2D element to transform from local to global coordinates.
        """
        pass

    @property
    @abstractmethod
    def local_stiffness_matrix(self) -> Matrix: ...

    @property
    @abstractmethod
    def local_forces_vector(self) -> Vector: ...

    @property
    @abstractmethod
    def type(self) -> _ElementType: ...

    @property
    @abstractmethod
    def geometry_properties(self) -> _GeometryProperties: ...

    @property
    @abstractmethod
    def container(self) -> Elements: ...

    @property
    @abstractmethod
    def index(self) -> int: ...

    @abstractmethod
    def set_elements_container(self, container: Elements) -> None: ...

    @abstractmethod
    def set_element_index(self, index: int) -> None: ...

    @property
    @abstractmethod
    def points(self) -> list[Point]: ...

    def __len__(self) -> int:
        return len(self._points)

    def print(self, idx: int | None = None) -> None:
        """
        This function prints the element information. If a argument idx is passed, so the info of the point is provided as (Point, Degrees of Freedom)
        """
        if not idx:
            print("\n" + self.__str__() + "\n")
        else:
            print(f"\n{self[idx]}\n")

    def __str__(self) -> str:
        begin = f"Element[{self.index:>3d}]:(\n  "
        element_data: list[str] = []

        for idx, point in enumerate(self._points):
            element_data.append(f"{idx + 1}: {repr(point)}")

        element_string = ",  ".join(element_data)

        end = f"\n  Number of interpolation points = {self.container.number_of_interpolation_points}\n  DOF = {self.container.dof_string(self.index)}\n)"
        return begin + element_string + end

    def __repr__(self) -> str:
        return f"Element({", ".join([repr(point) for point in self._points])})"

    def __getitem__(self, idx: int) -> tuple[Point, int | list[int]]:
        """
        The index is the point number, starting at 0, used in the list to create this element.

        Returns:
            tuple(Point, int | list[int]): A tuple with the point and the degree of freedom of the point
        """
        if idx > self.container.number_of_interpolation_points - 1:
            raise IndexError(
                f"The element doesn't have more than {self.container.number_of_interpolation_points} points. The index count starts at zero."
            )

        return (self._points[idx], self.container.degree_of_freedom(idx))
