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
