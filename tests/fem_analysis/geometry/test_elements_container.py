import pytest
from div_fem.fem_analysis.geometry.point import Point
from div_fem.fem_analysis.geometry.points import Points
from div_fem.fem_analysis.geometry.elements.element_2D import Element2D
from div_fem.fem_analysis.geometry.elements_container import Elements

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

def test_elements_singleton():
    elems1 = Elements()
    # Test early return in __init__ when already initialized
    elems1.__init__()
    assert elems1._initialized is True
    
    with pytest.raises(ValueError, match="The Elements object must be single per analysis."):
        elems2 = Elements()

def test_elements_add_and_retrieve():
    elems = Elements()
    pts = Points(2)
    p1 = Point([0.0, 0.0])
    p2 = Point([1.0, 0.0])
    p3 = Point([1.0, 1.0])
    pts.add([p1, p2, p3])
    
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    elem1 = Element2D((p1, p2), material_and_section_properties=mat)
    elem2 = Element2D((p2, p3), material_and_section_properties=mat)
    
    elems.add(elem1)
    elems.add([elem2])
    
    assert elems.number_of_elements == 2
    
    # Retrieval (1-based index)
    assert elems(1) == elem1
    assert elems(2) == elem2
    
    # Over limits
    with pytest.raises(IndexError, match="The element index is greater than the number of elements"):
        _ = elems[3]
        
    # Valid __getitem__
    assert elems[0] == elem1
    assert elems[1] == elem2

def test_elements_adjacency_and_rcm():
    elems = Elements()
    pts = Points(2)
    p1 = Point([0.0, 0.0])
    p2 = Point([1.0, 0.0])
    p3 = Point([1.0, 1.0])
    p4 = Point([0.0, 1.0])
    pts.add([p1, p2, p3, p4])
    
    # 1-2, 2-3, 3-4
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    e1 = Element2D((p1, p2), material_and_section_properties=mat)
    e2 = Element2D((p2, p3), material_and_section_properties=mat)
    e3 = Element2D((p3, p4), material_and_section_properties=mat)
    
    elems.add([e1, e2, e3])
    
    # adjacency should map 1<->2, 2<->3, 3<->4
    adj = elems.adjacency()
    assert 2 in adj[1]
    assert 1 in adj[2]
    assert 3 in adj[2]
    assert 4 in adj[3]
    
    # RCM
    rcm = elems.reverse_cuthill_mckee()
    # Path is basically a line, RCM will start at 1 or 4 and traverse.
    assert len(rcm) == 4
    assert rcm[0] in (1, 4)

def test_elements_str_print(capsys):
    elems = Elements()
    assert str(elems) == "Elements()"
    
    pts = Points(2)
    p1 = Point([0.0, 0.0])
    p2 = Point([1.0, 0.0])
    pts.add([p1, p2])
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    e1 = Element2D((p1, p2), material_and_section_properties=mat)
    elems(e1)
    
    assert "Element[  1]" in str(elems)
    
    elems.print()
    captured = capsys.readouterr()
    assert "Element[  1]" in captured.out

def test_elements_structural_analysis_descriptor():
    from unittest.mock import MagicMock
    elems = Elements()
    pts = Points(2)
    p1 = Point([0.0, 0.0])
    p2 = Point([1.0, 0.0])
    pts.add([p1, p2])
    
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    elem1 = Element2D((p1, p2), material_and_section_properties=mat)
    elems.add(elem1)
    
    elem1.interpolation_points_already_put_in_points_class = False
    
    dummy_points_class = MagicMock()
    
    class DummySA:
        @property
        def points(self):
            return dummy_points_class
    
    dummy_sa = DummySA()
    
    # Test __set__ descriptor
    elems.structural_analysis = dummy_sa
    
    assert elem1.interpolation_points_already_put_in_points_class is True
    dummy_points_class.assert_called_once()
    
    # Test exception block inside __set__
    class ErrorSA:
        @property
        def points(self):
            raise ValueError("Test Error")
            
    with pytest.raises(ValueError, match="Test Error"):
        elems.structural_analysis = ErrorSA()

def test_elements_string_for_element_info_coverage():
    elems = Elements()
    
    class MockElement1:
        index = 1
        extreme_points = ["P1", "P2"]
        interpolation_points = ["IP1", "IP2", "IP3"]
        
    class MockElement2:
        index = 2
        extreme_points = ["P1", "P2"]
        @property
        def interpolation_points(self):
            raise AttributeError("No points")
            
    class MockElement3:
        index = 3
        extreme_points = ["P1", "P2"]
        interpolation_points = ["IP1", "IP2"]
        
    elems._elements = [MockElement1(), MockElement2(), MockElement3()]
    
    str_repr = str(elems)
    # MockElement1 string formatting (len > 2)
    assert "'IP1',   ... ,  'IP3'" in str_repr
    
    # MockElement2 string formatting (exception block)
    assert "Element[  2]" in str_repr
    assert "'P1',  'P2'" in str_repr
    
    # MockElement3 string formatting (len <= 2)
    assert "Element[  3]" in str_repr
    assert "'IP1',  'IP2'" in str_repr
