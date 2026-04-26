import pytest
from div_fem.fem_analysis.geometry.point import Point
from div_fem.fem_analysis.geometry.points import Points
from div_fem.fem_analysis.geometry.elements.element_2D import Element2D
from div_fem.fem_analysis.geometry.elements_container import Elements
from div_fem.matrices.base_matrix import Matrix

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
