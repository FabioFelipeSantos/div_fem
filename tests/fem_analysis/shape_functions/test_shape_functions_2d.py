import pytest
import math
from div_fem.fem_analysis.shape_functions.shape_functions_2D import ShapeFunctions2D
from div_fem.matrices.base_matrix import Matrix
from div_fem.matrices.base_vector import Vector

class TestShapeFunctions2D:
    def test_initialization_bar(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=2, length=10.0, type="bar")
        assert sf.number_of_points == 2
        assert sf.jacobian == 5.0
        assert sf.nodal_inclination is None
        
        # Interpolation points for 2 points: [-1.0, 1.0]
        points = sf.interpolation_points.get_list()
        assert points == pytest.approx([-1.0, 1.0])
        
    def test_initialization_beam(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=4, length=10.0, type="beam")
        assert sf.number_of_points == 2
        assert sf.jacobian == 5.0
        assert sf.nodal_inclination is not None
        assert len(sf.nodal_inclination.get_list()) == 2
        
    def test_initialization_frame(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=6, length=10.0, type="frame")
        assert sf.number_of_points == 2
        assert sf.jacobian == 5.0
        assert sf.nodal_inclination is not None

    def test_bar_value_matrix(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=2, length=10.0, type="bar")
        val_minus_1 = sf.value(-1.0)
        assert isinstance(val_minus_1, Matrix)
        assert val_minus_1.get_list() == [[1.0, 0.0]]
        
        val_plus_1 = sf.value(1.0)
        assert val_plus_1.get_list() == [[0.0, 1.0]]
        
        val_zero = sf.value(0.0)
        assert val_zero.get_list() == [[0.5, 0.5]]

    def test_bar_value_index(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=2, length=10.0, type="bar")
        assert sf.value(-1.0, index=0) == 1.0
        assert sf.value(-1.0, index=1) == 0.0
        assert sf.value(0.0, index=0) == 0.5
        
        # List of indices
        assert sf.value(-1.0, index=[0, 1]) == [1.0, 0.0]

    def test_beam_value_matrix(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=4, length=10.0, type="beam")
        # 4 DOFs: [v1, theta1, v2, theta2]
        val_minus_1 = sf.value(-1.0)
        assert isinstance(val_minus_1, Matrix)
        assert len(val_minus_1.get_list()[0]) == 4
        # At xi=-1, N1=1, others=0
        assert val_minus_1.get_list()[0] == pytest.approx([1.0, 0.0, 0.0, 0.0])

        val_plus_1 = sf.value(1.0)
        assert val_plus_1.get_list()[0] == pytest.approx([0.0, 0.0, 1.0, 0.0])

    def test_beam_value_index(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=4, length=10.0, type="beam")
        assert sf.value(-1.0, index=0) == pytest.approx(1.0)
        assert sf.value(-1.0, index=1) == pytest.approx(0.0)
        assert sf.value(-1.0, index=[0, 2]) == pytest.approx([1.0, 0.0])

    def test_frame_value_matrix(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=6, length=10.0, type="frame")
        # 6 DOFs: [u1, v1, theta1, u2, v2, theta2]
        val_minus_1 = sf.value(-1.0)
        assert isinstance(val_minus_1, Matrix)
        assert len(val_minus_1.get_list()[0]) == 6
        assert val_minus_1.get_list()[0] == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    def test_frame_value_index(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=6, length=10.0, type="frame")
        assert sf.value(-1.0, index=0) == pytest.approx(1.0) # u1
        assert sf.value(-1.0, index=1) == pytest.approx(1.0) # v1
        assert sf.value(-1.0, index=2) == pytest.approx(0.0) # theta1
        assert sf.value(-1.0, index=[0, 3]) == pytest.approx([1.0, 0.0])
        # Covering indices with nodal_dof_zero_index != 0 in list for frame value()
        assert len(sf.value(-1.0, index=[1, 2, 4, 5])) == 4

    def test_bar_derivative_matrix(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=2, length=10.0, type="bar")
        der_0 = sf.derivative(1, 0.0)
        assert isinstance(der_0, Matrix)
        # N1 = (1-xi)/2 -> dN1/dxi = -0.5
        # N2 = (1+xi)/2 -> dN2/dxi = 0.5
        assert der_0.get_list()[0] == pytest.approx([-0.5, 0.5])

    def test_bar_derivative_index(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=2, length=10.0, type="bar")
        assert sf.derivative(1, 0.0, index=0) == pytest.approx(-0.5)
        assert sf.derivative(1, 0.0, index=[0, 1]) == pytest.approx([-0.5, 0.5])
        
        # Test second derivative (should be 0 for linear bar)
        assert sf.derivative(2, 0.0, index=0) == pytest.approx(0.0)
        
        # Test third derivative for bar (to cover diff_order=3 in _Lagrangian_derivative)
        # evaluated at a node and outside a node
        assert isinstance(sf.derivative(3, -1.0, index=0), float)
        assert isinstance(sf.derivative(3, 0.5, index=0), float)
        
        # Test derivatives at another node (to cover lines 344-360)
        assert isinstance(sf.derivative(1, -1.0, index=1), float)
        assert isinstance(sf.derivative(2, -1.0, index=1), float)
        assert isinstance(sf.derivative(3, -1.0, index=1), float)

    def test_beam_derivative_matrix(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=4, length=10.0, type="beam")
        der_0 = sf.derivative(1, 0.0)
        assert isinstance(der_0, Matrix)
        assert len(der_0.get_list()[0]) == 4

    def test_beam_derivative_index(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=4, length=10.0, type="beam")
        der_0_idx0 = sf.derivative(1, 0.0, index=0)
        assert isinstance(der_0_idx0, float)
        der_0_idx_list = sf.derivative(1, 0.0, index=[0, 1])
        assert isinstance(der_0_idx_list, list)
        assert len(der_0_idx_list) == 2
        
        # Test diff_order 2 and 3 to cover lines 409-441
        der_2 = sf.derivative(2, 0.0)
        assert isinstance(der_2, Matrix)
        der_3 = sf.derivative(3, 0.0)
        assert isinstance(der_3, Matrix)

    def test_frame_derivative_matrix(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=6, length=10.0, type="frame")
        der_0 = sf.derivative(1, 0.0)
        assert isinstance(der_0, Matrix)
        assert len(der_0.get_list()[0]) == 6

    def test_frame_derivative_index(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=6, length=10.0, type="frame")
        der_0_idx0 = sf.derivative(1, 0.0, index=0)
        assert isinstance(der_0_idx0, float)
        
        # Cover single index with nodal_dof_zero_index != 0
        der_0_idx1 = sf.derivative(1, 0.0, index=1)
        assert isinstance(der_0_idx1, float)
        der_0_idx2 = sf.derivative(1, 0.0, index=2)
        assert isinstance(der_0_idx2, float)
        
        der_0_idx_list = sf.derivative(1, 0.0, index=[0, 1, 2, 3])
        assert isinstance(der_0_idx_list, list)
        assert len(der_0_idx_list) == 4

    def test_invalid_derivative_order(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=2, length=10.0, type="bar")
        with pytest.raises(ValueError, match="derivative of shape functions is only available for orders"):
            sf.derivative(4, 0.0)

    def test_invalid_index(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=2, length=10.0, type="bar")
        with pytest.raises(IndexError, match="expected at maximum of 1"):
            sf.value(0.0, index=2)
            
        with pytest.raises(IndexError, match="expected at maximum of 1"):
            sf.derivative(1, 0.0, index=[0, 2])

    def test_missing_nodal_inclination(self):
        sf = ShapeFunctions2D(number_of_points=2, total_degree_of_freedom=4, length=10.0, type="beam")
        sf._nodal_inclination = None
        
        with pytest.raises(ValueError, match="To calculate Hermite functions, the nodal inclination must be calculated before."):
            sf.value(0.0)
            
        with pytest.raises(ValueError, match="To calculate Hermite functions, the nodal inclination must be calculated before."):
            sf.derivative(1, 0.0)
