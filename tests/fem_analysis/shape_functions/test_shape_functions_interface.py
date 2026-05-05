import pytest
from div_fem.fem_analysis.geometry.elements.shape_functions_interface import ShapeFunctions
from div_fem.matrices.base_matrix import Matrix
from div_fem.matrices.base_vector import Vector


class DummyShapeFunctions(ShapeFunctions):
    def __init__(self):
        self._number_of_points = 2
        self._interpolation_points = Vector([-1.0, 1.0])
        self._barycentric_weights = Vector([0.5, 0.5])
        self._jacobian = 1.0
        self._nodal_inclination = None

    def value(self, xi, index=None):
        pass

    def derivative(self, diff_order, xi, index=None, *, is_for_stiffness_matrix=False):
        pass

    @property
    def number_of_points(self):
        return self._number_of_points

    @property
    def interpolation_points(self):
        return self._interpolation_points

    @property
    def barycentric_weights(self):
        return self._barycentric_weights

    @property
    def nodal_inclination(self):
        return self._nodal_inclination

    @property
    def jacobian(self):
        return super().jacobian


def test_shape_functions_interface_instantiation():
    # Attempting to instantiate the abstract class directly should fail
    with pytest.raises(TypeError):
        ShapeFunctions()


def test_dummy_shape_functions():
    dummy = DummyShapeFunctions()
    
    assert dummy.number_of_points == 2
    assert dummy.interpolation_points.get_list() == [-1.0, 1.0]
    assert dummy.barycentric_weights.get_list() == [0.5, 0.5]
    assert dummy.jacobian == 1.0
    assert dummy.nodal_inclination is None
    
    # Test __str__ method
    str_repr = str(dummy)
    assert "ShapeFunctions(" in str_repr
    assert "number_of_points=2" in str_repr
    assert "points=" in str_repr
