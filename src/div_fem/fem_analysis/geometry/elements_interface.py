from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Mapping, TypeVar

from div_fem.fem_analysis.geometry.elements.element_load_interface import (
    ElementLoadInterface,
)
from div_fem.fem_analysis.geometry.elements.shape_functions_interface import (
    ShapeFunctions,
)
from div_fem.fem_analysis.geometry.point import Point
from div_fem.matrices.base_matrix import Matrix
from div_fem.matrices.base_vector import Vector
from div_fem.utils.descriptors.descriptor_private_name import DescriptorBaseClass

if TYPE_CHECKING:
    from div_fem.fem_analysis.geometry.elements_container import Elements

_GeometryProperties = TypeVar("_GeometryProperties", bound=Mapping[str, Any], covariant=True)
_MaterialAndSectionProperties = TypeVar("_MaterialAndSectionProperties", bound=Mapping[str, Any], covariant=True)
_ElementType = TypeVar("_ElementType", bound=str, covariant=True)
_ElementLoad = TypeVar("_ElementLoad", bound=ElementLoadInterface, covariant=True)


def index_validation(value: int) -> None:
    if value < 0:
        raise ValueError(f"A index for the element must be greater than 0. Received {value}")

    return


class ElementInterface(
    ABC,
    Generic[
        _ElementType,
        _GeometryProperties,
        _MaterialAndSectionProperties,
        _ElementLoad,
    ],
):
    extreme_points: tuple[Point, Point]
    interpolation_points: list[Point]
    geometry_properties: _GeometryProperties
    material_and_section_properties: _MaterialAndSectionProperties
    _loads: list[_ElementLoad] | None

    type = DescriptorBaseClass[_ElementType]()
    number_interpolation_points = DescriptorBaseClass[int]()
    interpolation_points_already_put_in_points_class: bool

    elements_container = DescriptorBaseClass["Elements"]()
    index = DescriptorBaseClass[int](validation=index_validation)
    total_degree_of_freedom = DescriptorBaseClass[int]()
    shape_functions = DescriptorBaseClass[ShapeFunctions]()

    @property
    @abstractmethod
    def points(self) -> list[Point]: ...

    @property
    @abstractmethod
    def degree_of_freedom(self) -> list[int]: ...

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

    def print(self, idx: int | None = None) -> None:
        """
        This function prints the element information. If a argument idx is passed, so the info of the point is provided as (Point, Degrees of Freedom)
        """
        if not idx:
            print("\n" + self.__str__() + "\n")
        else:
            print(f"\n{self[idx]}\n")

    def __str__(self) -> str:
        begin = f"Element(\n  "
        element_data: list[str] = []

        for idx, point in enumerate(self.extreme_points):
            dof_numbers: list[int] | None = getattr(point, "dof_numbers", None)
            if not dof_numbers:
                element_data.append(f"{idx + 1}: {repr(point)}")
            else:
                element_data.append(f"{idx + 1}: {repr(point)}, DOF: {dof_numbers}")

        element_string = ";  ".join(element_data)

        end = f"\n  Number of interpolation points = {self.number_interpolation_points}\n)"
        return begin + element_string + end

    def __repr__(self) -> str:
        return f"Element({", ".join([repr(point) for point in self.extreme_points])})"

    def __getitem__(self, idx: int) -> tuple[Point, list[int]]:
        """
        The index is the point number, starting at 0, used in the list to create this element.

        Returns:
            tuple(Point, int | list[int]): A tuple with the point and the degree of freedom of the point
        """
        if idx > self.number_interpolation_points - 1:
            raise IndexError(
                f"The element doesn't have more than {self.number_interpolation_points} points. The index count starts at zero."
            )

        return (self.extreme_points[idx], self.extreme_points[idx].dof_numbers)
