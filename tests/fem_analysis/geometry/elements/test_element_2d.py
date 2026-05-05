import pytest
from div_fem.fem_analysis.geometry.point import Point
from div_fem.fem_analysis.geometry.points import Points
from div_fem.fem_analysis.geometry.elements.element_2D import Element2D
from div_fem.fem_analysis.geometry.elements_container import Elements
from div_fem.matrices.base_matrix import Matrix
from div_fem.fem_analysis.loads.element_2D_loads import Element2DLoads

@pytest.fixture(autouse=True)
def reset_singletons():
    # Reset Points
    Points._instance = None
    Points._initialized = False
    Points._points = []
    Points._index_next_point = 1
    # Reset Elements
    Elements._instance = None
    Elements._initialized = False
    Elements._elements = []
    Elements._index_next_element = 1
    yield
    Points._instance = None
    Points._initialized = False
    Points._points = []
    Points._index_next_point = 1
    Elements._instance = None
    Elements._initialized = False
    Elements._elements = []
    Elements._index_next_element = 1

def test_element_2d_init_and_geometry():
    # 3-4-5 triangle to have exact cosine values
    p1 = Point([0.0, 0.0])
    p2 = Point([4.0, 3.0])
    pts = Points(2)
    pts.add([p1, p2])
    
    mat = {"E": 200e9, "A": 0.01, "I": 0.0001}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="bar")
    
    assert elem.geometry_properties["length"] == 5.0
    assert elem.geometry_properties["cosine_x"] == 0.8
    assert elem.geometry_properties["cosine_y"] == 0.6
    
    assert elem.total_degree_of_freedom == 4
    assert isinstance(elem.T, Matrix)
    
    # Points include the extreme and interpolation points
    assert len(elem.points) == 2

def test_element_2d_str_and_repr(capsys):
    p1 = Point([0.0, 0.0])
    p2 = Point([1.0, 0.0])
    pts = Points(2)
    pts.add([p1, p2])
    
    mat = {"E": 200e9, "A": 0.01, "I": 0.0001}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="bar")
    
    # Repr
    assert "Element(Point" in repr(elem)
    
    # Str and print
    elem.print()
    captured = capsys.readouterr()
    assert "Element(" in captured.out
    assert "Number of interpolation points = 2" in captured.out

def test_element_2d_stiffness_matrix_errors():
    p1 = Point([0.0, 0.0])
    p2 = Point([1.0, 0.0])
    p1.dof_per_node = 2
    p2.dof_per_node = 2
    
    # Force dict to miss keys for validation logic
    mat_no_E = {"A": 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat_no_E, type="bar") # type: ignore
    with pytest.raises(ValueError, match="The material parameter E \\(Youngs Modulus\\) must be passed to integration function."):
        _ = elem.local_stiffness_matrix

    mat_no_A = {"E": 1.0, "I": 1.0}
    elem2 = Element2D((p1, p2), material_and_section_properties=mat_no_A, type="bar") # type: ignore
    with pytest.raises(ValueError, match="The area for the cross section must be provided in the integration function."):
        _ = elem2.local_stiffness_matrix

    mat_no_I = {"E": 1.0, "A": 1.0}
    elem3 = Element2D((p1, p2), material_and_section_properties=mat_no_I, type="beam") # type: ignore
    with pytest.raises(ValueError, match="The moment of inertia for the cross section must be provided in the integration function."):
        _ = elem3.local_stiffness_matrix

def test_element_2d_forces_vector_errors():
    p1 = Point([0.0, 0.0])
    p2 = Point([1.0, 0.0])
    p1.dof_per_node = 2
    p2.dof_per_node = 2
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="bar") # type: ignore
    with pytest.raises(ValueError, match="The element doesn't have any loads applied."):
        _ = elem.local_forces_vector

def test_element_2d_getitem():
    p1 = Point([0.0, 0.0])
    p2 = Point([1.0, 0.0])
    pts = Points(2)
    pts.add([p1, p2])
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="bar")
    
    point_tuple = elem[0]
    assert point_tuple[0] == p1
    assert point_tuple[1] == [0, 1]
    
    with pytest.raises(IndexError, match="The element doesn't have more than"):
        _ = elem[2]

def test_element_2d_bar_stiffness_matrix():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(1)
    pts.add([p1, p2])
    
    mat = {"E": 200e9, "A": 0.01, "I": 0.0001}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="bar")
    
    # Exact analytical matrix: EA/L = (200e9 * 0.01) / 10 = 2e8
    expected = 2e8
    k = elem.local_stiffness_matrix
    assert pytest.approx(k[0, 0]) == expected
    assert pytest.approx(k[0, 1]) == -expected
    assert pytest.approx(k[1, 0]) == -expected
    assert pytest.approx(k[1, 1]) == expected

def test_element_2d_bar_forces():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(1)
    pts.add([p1, p2])
    
    mat = {"E": 200e9, "A": 0.01, "I": 0.0001}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="bar")
    
    # Concentrated force at x=0.5 (middle) -> should distribute evenly
    load1 = Element2DLoads(type="concentrated", force_value_x=1000.0, force_point=0.5)
    elem._loads = [load1]
    f = elem.local_forces_vector
    assert pytest.approx(f[0]) == 500.0
    assert pytest.approx(f[1]) == 500.0
    elem._loads = [] # Reset loads
    
    # Constant load q_x = 100
    load2 = Element2DLoads(type="constant", force_value_x=100.0)
    elem._loads = [load2]
    f2 = elem.local_forces_vector
    # Integral of constant q_x is q_x * L / 2 for each node
    assert pytest.approx(f2[0]) == 500.0
    assert pytest.approx(f2[1]) == 500.0
    elem._loads = []
    
    # Function load q_x(x) = x -> linear increasing from 0 to 1
    # L/6 and L/3 -> 10/6 and 10/3
    load3 = Element2DLoads(type="function", force_value_x=lambda x: x)
    elem._loads = [load3]
    f3 = elem.local_forces_vector
    assert pytest.approx(f3[0]) == 10.0 / 6.0
    assert pytest.approx(f3[1]) == 10.0 / 3.0

def test_element_2d_bar_forces_errors():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(1)
    pts.add([p1, p2])
    
    mat = {"E": 200e9, "A": 0.01, "I": 0.0001}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="bar")
    
    load_err = Element2DLoads(type="concentrated", force_value_y=1000.0, force_point=0.5)
    elem._loads = [load_err]
    
    with pytest.raises(ValueError, match="The value for a concentrated force or moment must be a valid float number."):
        _ = elem.local_forces_vector

def test_element_2d_beam_stiffness_matrix():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(2)
    pts.add([p1, p2])
    
    mat = {"E": 200e9, "A": 0.01, "I": 0.0001}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="beam")
    
    # Exact analytical matrix for beam: EI/L^3 * [12, 6L, -12, 6L; 6L, 4L^2, -6L, 2L^2; -12, -6L, 12, -6L; 6L, 2L^2, -6L, 4L^2]
    # EI/L^3 = (200e9 * 0.0001) / 1000 = 2e7 / 1000 = 20000
    expected = [
        [240000.0, 1200000.0, -240000.0, 1200000.0],
        [1200000.0, 8000000.0, -1200000.0, 4000000.0],
        [-240000.0, -1200000.0, 240000.0, -1200000.0],
        [1200000.0, 4000000.0, -1200000.0, 8000000.0]
    ]
    
    k = elem.local_stiffness_matrix
    for i in range(4):
        for j in range(4):
            assert pytest.approx(k[i, j]) == expected[i][j]

def test_element_2d_beam_forces():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(2)
    pts.add([p1, p2])
    
    mat = {"E": 200e9, "A": 0.01, "I": 0.0001}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="beam")
    
    # Concentrated force Fy=-1000 at x=0.5
    load1 = Element2DLoads(type="concentrated", force_value_y=-1000.0, force_point=0.5)
    elem._loads = [load1]
    f1 = elem.local_forces_vector
    assert pytest.approx(f1[0]) == -500.0
    assert pytest.approx(f1[1]) == -1250.0
    assert pytest.approx(f1[2]) == -500.0
    assert pytest.approx(f1[3]) == 1250.0
    
    # Constant load qy=-100
    load2 = Element2DLoads(type="constant", force_value_y=-100.0)
    elem._loads = [load2]
    f2 = elem.local_forces_vector
    assert pytest.approx(f2[0]) == -500.0
    assert pytest.approx(f2[1]) == -100.0 * 100.0 / 12.0
    assert pytest.approx(f2[2]) == -500.0
    assert pytest.approx(f2[3]) == 100.0 * 100.0 / 12.0

def test_element_2d_beam_forces_errors():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(2)
    pts.add([p1, p2])
    
    mat = {"E": 200e9, "A": 0.01, "I": 0.0001}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="beam")
    
    load_err = Element2DLoads(type="concentrated", force_value_x=1000.0, force_point=0.5)
    elem._loads = [load_err]
    
def test_element_2d_frame_stiffness_matrix():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(3)
    pts.add([p1, p2])
    
    mat = {"E": 200e9, "A": 0.01, "I": 0.0001}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="frame")
    
    # Superposition of Bar (2e8) and Beam (EI/L^3 matrix)
    k = elem.local_stiffness_matrix
    assert k.rows == 6 and k.columns == 6
    
    # Bar components
    assert pytest.approx(k[0, 0]) == 2e8
    assert pytest.approx(k[0, 3]) == -2e8
    assert pytest.approx(k[3, 0]) == -2e8
    assert pytest.approx(k[3, 3]) == 2e8
    
    # Beam components
    assert pytest.approx(k[1, 1]) == 240000.0
    assert pytest.approx(k[1, 2]) == 1200000.0
    assert pytest.approx(k[2, 5]) == 4000000.0
    
    # Ensure orthogonality (no coupling between axial and bending in local stiffness)
    assert pytest.approx(k[0, 1]) == 0.0
    assert pytest.approx(k[1, 0]) == 0.0

def test_element_2d_frame_t_matrix():
    # 3-4-5 triangle -> cos_x = 0.8, cos_y = 0.6
    p1 = Point([0.0, 0.0])
    p2 = Point([4.0, 3.0])
    pts = Points(3)
    pts.add([p1, p2])
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="frame")
    
    T = elem.T
    assert T.rows == 6 and T.columns == 6
    
    # Upper left 3x3 block
    assert pytest.approx(T[0, 0]) == 0.8
    assert pytest.approx(T[0, 1]) == 0.6
    assert pytest.approx(T[1, 0]) == -0.6
    assert pytest.approx(T[1, 1]) == 0.8
    assert pytest.approx(T[2, 2]) == 1.0
    
    # Lower right 3x3 block
    assert pytest.approx(T[3, 3]) == 0.8
    assert pytest.approx(T[3, 4]) == 0.6
    assert pytest.approx(T[4, 3]) == -0.6
    assert pytest.approx(T[4, 4]) == 0.8
    assert pytest.approx(T[5, 5]) == 1.0

def test_element_2d_frame_forces():
    # Inclined element: 3-4-5 triangle (length 5)
    p1 = Point([0.0, 0.0])
    p2 = Point([4.0, 3.0])
    pts = Points(3)
    pts.add([p1, p2])
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="frame")
    
    # Concentrated force Global Fx = 1000 at middle
    # Local: F_axial = 1000*0.8 = 800. F_perp = 1000*(-0.6) = -600.
    load1 = Element2DLoads(type="concentrated", force_value_x=1000.0, force_point=0.5)
    elem._loads = [load1]
    f = elem.local_forces_vector
    
    # Axial distribution: 800 / 2 = 400
    assert pytest.approx(f[0]) == 400.0
    assert pytest.approx(f[3]) == 400.0
    
    # Perpendicular distribution: -600 / 2 = -300
    assert pytest.approx(f[1]) == -300.0
    assert pytest.approx(f[4]) == -300.0
    
    # Moment distribution: P*L/8 = -600 * 5 / 8 = -375 (M1), and +375 (M2)
    assert pytest.approx(f[2]) == -375.0
    assert pytest.approx(f[5]) == 375.0

def test_element_2d_frame_forces_errors():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(3)
    pts.add([p1, p2])
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="frame")
    
    # Test moment without value
    load_err = Element2DLoads(type="moment", force_value_x=1.0, force_point=0.5)
    elem._loads = [load_err]
    
    with pytest.raises(ValueError, match="To concentrated moment in frames, provide a float value for the moment."):
        _ = elem.local_forces_vector

from unittest.mock import MagicMock

def test_element_2d_properties():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(2)
    pts.add([p1, p2])
    mat = {"E": 200e9, "A": 0.01, "I": 0.0001}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="beam")
    
    assert elem.points == [p1, p2]
    assert elem.degree_of_freedom == [0, 1, 2, 3]

def test_element_2d_integration_errors_bar():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(1)
    pts.add([p1, p2])
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="bar")
    
    mock_load = MagicMock(spec=Element2DLoads)
    mock_load.force_point = None
    mock_load.force_init_point = None
    mock_load.force_final_point = None
    
    # Missing force_x
    mock_load.force_type = "constant"
    mock_load.force_value.return_value = None
    elem._loads = [mock_load]
    with pytest.raises(ValueError, match="To apply loads on an bar element, provide a float or Callable"):
        _ = elem.local_forces_vector
        
    # Constant not float
    mock_load.force_value.return_value = "not float"
    with pytest.raises(ValueError, match="Provide a valid float for constant distributed load value for the x axis for elements of type bar."):
        _ = elem.local_forces_vector
        
    # Function not callable
    mock_load.force_type = "function"
    with pytest.raises(ValueError, match="For a function force in a bar, provide a callable"):
        _ = elem.local_forces_vector

def test_element_2d_integration_errors_beam():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(2)
    pts.add([p1, p2])
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="beam")
    
    mock_load = MagicMock(spec=Element2DLoads)
    mock_load.force_point = None
    mock_load.force_init_point = None
    mock_load.force_final_point = None
    
    # Missing force_y and moment
    mock_load.force_type = "constant"
    mock_load.force_value.return_value = (None, None)
    elem._loads = [mock_load]
    with pytest.raises(ValueError, match="To apply loads on an bar element, provide a float or Callable"):
        _ = elem.local_forces_vector
        
    # Constant force_y not float
    mock_load.force_value.return_value = ("not float", None)
    with pytest.raises(ValueError, match="Provide a valid float for constant load values for the y axis for elements of type beam."):
        _ = elem.local_forces_vector
        
    # Constant moment not float
    mock_load.force_value.return_value = (None, "not float")
    with pytest.raises(ValueError, match="Provide a valid float for constant load values for moment for elements of type beam."):
        _ = elem.local_forces_vector
        
    # Function force_y not callable
    mock_load.force_type = "function"
    mock_load.force_value.return_value = ("not callable", None)
    with pytest.raises(ValueError, match="For a function force in y axis, provide a callable"):
        _ = elem.local_forces_vector
        
    # Function moment not callable
    mock_load.force_value.return_value = (None, "not callable")
    with pytest.raises(ValueError, match="For a function force to moment, provide a callable"):
        _ = elem.local_forces_vector

def test_element_2d_integration_errors_frame():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(3)
    pts.add([p1, p2])
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="frame")
    
    mock_load = MagicMock(spec=Element2DLoads)
    mock_load.force_point = None
    mock_load.force_init_point = None
    mock_load.force_final_point = None
    
    # Missing all
    mock_load.force_type = "constant"
    mock_load.force_value.return_value = (None, None, None)
    elem._loads = [mock_load]
    with pytest.raises(ValueError, match="To apply loads on an frame element, provide a float or Callable"):
        _ = elem.local_forces_vector
        
    # Constant not float
    mock_load.force_value.return_value = ("not float", None, None)
    with pytest.raises(ValueError, match="Provide a valid float for constant load values for the x axis for elements of type frame."):
        _ = elem.local_forces_vector
    mock_load.force_value.return_value = (None, "not float", None)
    with pytest.raises(ValueError, match="Provide a valid float for constant load values for the y axis for elements of type frame."):
        _ = elem.local_forces_vector
    mock_load.force_value.return_value = (None, None, "not float")
    with pytest.raises(ValueError, match="Provide a valid float for constant load values for moment for elements of type frame."):
        _ = elem.local_forces_vector
        
    # Function not callable
    mock_load.force_type = "function"
    mock_load.force_value.return_value = ("not callable", None, None)
    with pytest.raises(ValueError, match="Provide a valid Callable"):
        _ = elem.local_forces_vector
    mock_load.force_value.return_value = (None, "not callable", None)
    with pytest.raises(ValueError, match="Provide a valid Callable.*y axis.*frame"):
        _ = elem.local_forces_vector
    mock_load.force_value.return_value = (None, None, "not callable")
    with pytest.raises(ValueError, match="Provide a valid Callable.*moment.*frame"):
        _ = elem.local_forces_vector

def test_element_2d_concentrated_errors():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(3)
    pts.add([p1, p2])
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    
    # Beam concentrated errors
    elem_beam = Element2D((p1, p2), material_and_section_properties=mat, type="beam")
    mock_load = MagicMock(spec=Element2DLoads)
    mock_load.force_type = "concentrated"
    mock_load.force_point = 0.5
    
    mock_load.force_value.return_value = (None, None)
    elem_beam._loads = [mock_load]
    with pytest.raises(ValueError, match="To concentrated loads in beams, provide a value for y axis or for moment."):
        _ = elem_beam.local_forces_vector
        
    mock_load.force_type = "moment"
    mock_load.force_value.return_value = (1.0, None)
    with pytest.raises(ValueError, match="To concentrated moment in beams, provide a float value for the moment."):
        _ = elem_beam.local_forces_vector
        
    mock_load.force_type = "concentrated"
    mock_load.force_value.return_value = ("not float", None)
    with pytest.raises(ValueError, match="To concentrated y axis load provide a float value."):
        _ = elem_beam.local_forces_vector
        
    mock_load.force_value.return_value = (None, "not float")
    with pytest.raises(ValueError, match="To concentrated y axis load provide a float value."):
        _ = elem_beam.local_forces_vector

    # Frame concentrated errors
    elem_frame = Element2D((p1, p2), material_and_section_properties=mat, type="frame")
    mock_load.force_type = "concentrated"
    mock_load.force_value.return_value = (None, None, None)
    elem_frame._loads = [mock_load]
    with pytest.raises(ValueError, match="To concentrated loads in frames, provide a value for x or y axis or for moment."):
        _ = elem_frame.local_forces_vector
        
    mock_load.force_type = "moment"
    mock_load.force_value.return_value = (1.0, None, None)
    with pytest.raises(ValueError, match="To concentrated moment in frames, provide a float value for the moment."):
        _ = elem_frame.local_forces_vector



def test_element_2d_stiffness_integration_errors():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(3)
    pts.add([p1, p2])
    mat = {'E': 1.0, 'A': 1.0, 'I': 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type='frame')
    
    # Missing E
    with pytest.raises(ValueError, match='Youngs Modulus.*passed to integration function'):
        elem._integration_function_for_stiffness_matrix(0.0, L=10.0, A=1.0, I=1.0)
        
    # Missing L
    with pytest.raises(ValueError, match='length of the bar must be passed to integration function'):
        elem._integration_function_for_stiffness_matrix(0.0, E=1.0, A=1.0, I=1.0)
        
    # Missing A for frame
    with pytest.raises(ValueError, match='area.*provided in the integration function'):
        elem._integration_function_for_stiffness_matrix(0.0, E=1.0, L=10.0, I=1.0)
        
    # Missing I for frame
    with pytest.raises(ValueError, match='moment of inertia.*provided in the integration function'):
        elem._integration_function_for_stiffness_matrix(0.0, E=1.0, L=10.0, A=1.0)

def test_element_2d_integration_function_load_errors():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(1)
    pts.add([p1, p2])
    elem = Element2D((p1, p2), material_and_section_properties={'E':1, 'A':1}, type='bar')
    
    # Missing load
    with pytest.raises(ValueError, match='Some force must be provided'):
        elem._integration_function_for_forces_vector(0.0)
        
    # Not an Element2DLoads
    with pytest.raises(ValueError, match='must be a valid Element2DLoads class'):
        elem._integration_function_for_forces_vector(0.0, load='not a load')
        
    mock_load = MagicMock(spec=Element2DLoads)
    # Missing L
    with pytest.raises(ValueError, match='length of the bar must be passed'):
        elem._integration_function_for_forces_vector(0.0, load=mock_load)
        
    # L not float
    with pytest.raises(ValueError, match='length of the element must be a float'):
        elem._integration_function_for_forces_vector(0.0, load=mock_load, L='not float')

def test_element_2d_calculating_points_try_except():
    # This hits the 'pass' in the try/except of _calculating_points
    # if self.elements_container.structural_analysis.points(interpolation_points) fails
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(2)
    pts.add([p1, p2])
    
    elem = Element2D((p1, p2), material_and_section_properties={'E':1, 'A':1, 'I':1}, type='beam')
    # We don't have a structural_analysis set up in the mock elements_container usually, 
    # so it should already be hitting the pass.
    # To be sure, we can check the flag.
    assert elem.interpolation_points_already_put_in_points_class == False



def test_element_2d_points_interpolation():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(2)
    pts.add([p1, p2])
    # With 3 interpolation points
    elem = Element2D((p1, p2), material_and_section_properties={'E':1, 'A':1}, type='bar', number_interpolation_points=3)
    assert len(elem.points) == 3

def test_element_2d_stiffness_integration_problem():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(2)
    pts.add([p1, p2])
    elem = Element2D((p1, p2), material_and_section_properties={'E':1, 'A':1}, type='bar')
    elem.type = 'invalid'
    with pytest.raises(ValueError, match='Some problem in the definition of your functions to integrate'):
        elem._integration_function_for_stiffness_matrix(0.0, E=1.0, L=10.0, A=1.0, I=1.0)



def test_element_2d_frame_distributed_forces():
    p1 = Point([0.0, 0.0])
    p1.dof_per_node = 3
    p2 = Point([10.0, 0.0])
    p2.dof_per_node = 3
    pts = Points(3)
    pts.add([p1, p2])
    mat = {'E': 1.0, 'A': 1.0, 'I': 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type='frame')
    
    # Constant load: qx=10, qy=20, m=30
    load = Element2DLoads(type='constant', force_value_x=10.0, force_value_y=20.0, force_value_moment=30.0)
    elem._loads = [load]
    f = elem.local_forces_vector
    assert f.rows == 6
    
    # Function load frame
    load_func = Element2DLoads(type='function', force_value_x=lambda x: 10.0, force_value_y=lambda x: 20.0, force_value_moment=lambda x: 30.0)
    elem._loads = [load_func]
    f_func = elem.local_forces_vector
    assert f_func.rows == 6



def test_element_2d_bar_distributed_forces():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(1)
    pts.add([p1, p2])
    mat = {'E': 1.0, 'A': 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type='bar')
    
    # Constant load qx=10
    load = Element2DLoads(type='constant', force_value_x=10.0)
    elem._loads = [load]
    f = elem.local_forces_vector
    assert pytest.approx(f[0]) == 50.0
    assert pytest.approx(f[1]) == 50.0
    
    # Function load qx = 10*x
    load_func = Element2DLoads(type='function', force_value_x=lambda x: 10.0*x)
    elem._loads = [load_func]
    f_func = elem.local_forces_vector
    assert f_func.rows == 2

def test_element_2d_beam_function_forces():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(2)
    pts.add([p1, p2])
    mat = {'E': 1.0, 'A': 1.0, 'I': 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type='beam')
    
    # Function load qy=20, m=30
    load = Element2DLoads(type='function', force_value_y=lambda x: 20.0, force_value_moment=lambda x: 30.0)
    elem._loads = [load]
    f = elem.local_forces_vector
    assert f.rows == 4
    
    # Constant moment
    load_m = Element2DLoads(type='constant', force_value_moment=30.0)
    elem._loads = [load_m]
    f_m = elem.local_forces_vector
    assert f_m.rows == 4



def test_element_2d_missing_force_point():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(1)
    pts.add([p1, p2])
    elem = Element2D((p1, p2), material_and_section_properties={'E':1, 'A':1}, type='bar')
    # Element2DLoads validates force_point at construction for concentrated loads,
    # so create with a valid point then null it to test the Element2D-level check
    load = Element2DLoads(type='concentrated', force_value_x=10.0, force_point=0.5)
    load._force_point = None
    elem._loads = [load]
    with pytest.raises(ValueError, match='concentrated forces a float value'):
        _ = elem.local_forces_vector



from div_fem.fem_analysis.loads.element_2D_loads import Element2DLoads

def test_element_2d_loads_extra():
    # Missing all values
    with pytest.raises(ValueError, match='at least one type of value'):
        Element2DLoads(type='constant')
    
    # Missing force_point for concentrated
    with pytest.raises(ValueError, match='must be provided with a local x coordinate'):
        Element2DLoads(type='concentrated', force_value_x=10.0)
    
    # force_point provided for constant
    with pytest.raises(ValueError, match='non concentrated force cannot be provided with a point of application'):
        Element2DLoads(type='constant', force_value_x=10.0, force_point=0.5)

    # __str__ coverage
    load = Element2DLoads(type='concentrated', force_value_x=10.0, force_point=0.5)
    assert 'Force(concentrated' in str(load)
    
    load_func = Element2DLoads(type='function', force_value_x=lambda x: x)
    assert 'Force(function' in str(load_func)
    
    # Hit unreachable line 167 in element_2D.py by manual override
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(2)
    pts.add([p1, p2])
    elem = Element2D((p1, p2), material_and_section_properties={'E':1, 'A':1}, type='bar')
    load_hack = Element2DLoads(type='concentrated', force_value_x=10.0, force_point=0.5)
    load_hack._force_point = None
    elem._loads = [load_hack]
    with pytest.raises(ValueError, match='concentrated forces a float value'):
        _ = elem.local_forces_vector



def test_element_2d_interpolation_points_flag():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(1)
    pts.add([p1, p2])
    mock_container = MagicMock()
    mock_container.structural_analysis = MagicMock()
    elem = Element2D((p1, p2), material_and_section_properties={'E':1, 'A':1}, type='bar')
    elem.elements_container = mock_container
    elem._calculating_points()
    assert elem.interpolation_points_already_put_in_points_class == True

def test_element_2d_beam_concentrated_moment_hit():
    p1 = Point([0.0, 0.0])
    p1.dof_per_node = 2
    p2 = Point([10.0, 0.0])
    p2.dof_per_node = 2
    pts = Points(2)
    pts.add([p1, p2])
    mat = {'E': 1.0, 'A': 1.0, 'I': 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type='beam')
    load = Element2DLoads(type='moment', force_value_moment=100.0, force_point=0.5)
    elem._loads = [load]
    f = elem.local_forces_vector
    assert f.rows == 4

def test_element_2d_frame_concentrated_all_hit():
    p1 = Point([0.0, 0.0])
    p1.dof_per_node = 3
    p2 = Point([10.0, 0.0])
    p2.dof_per_node = 3
    pts = Points(3)
    pts.add([p1, p2])
    mat = {'E': 1.0, 'A': 1.0, 'I': 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type='frame')
    load = Element2DLoads(type='concentrated', force_value_x=10.0, force_value_y=20.0, force_value_moment=30.0, force_point=0.5)
    elem._loads = [load]
    f = elem.local_forces_vector
    assert f.rows == 6



def test_element_2d_properties_final():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(1)
    pts.add([p1, p2])

    elem = Element2D((p1, p2), material_and_section_properties={'E':1, 'A':1}, type='bar')

    # elem.points returns a list: [extreme_0, *interpolation, extreme_1]
    # For 2 interpolation points and no interior nodes, it's [p1, p2]
    all_points = elem.points
    assert isinstance(all_points, list)
    assert all_points[0] is p1
    assert all_points[-1] is p2
    # degree_of_freedom is a flat list of all DOF numbers
    assert elem.degree_of_freedom == p1.dof_numbers + p2.dof_numbers
