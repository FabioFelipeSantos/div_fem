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

def test_element_2d_dof_errors():
    p1 = Point([0.0, 0.0])
    p2 = Point([1.0, 0.0])
    p1.dof_per_node = 2
    p2.dof_per_node = 2
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="bar")
    
    with pytest.raises(ValueError, match="Provide just one information for degree of freedom for each node. Received 2 nodes and 1 DOF info."):
        elem._verifying_dof_number(2, [[1]])
        
    with pytest.raises(ValueError, match="To an element of type bar, each point must have just one degree of freedom."):
        elem._verifying_dof_number(2, [[1, 2], [3, 4]])

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
