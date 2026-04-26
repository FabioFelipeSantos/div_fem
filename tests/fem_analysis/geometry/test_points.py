import pytest
from div_fem.fem_analysis.geometry.point import Point
from div_fem.fem_analysis.geometry.points import Points

@pytest.fixture(autouse=True)
def reset_singletons():
    # Reset Points singleton before each test
    Points._instance = None
    Points._initialized = False
    Points._points = []
    Points._index_next_point = 1
    yield
    # Reset again just to be safe
    Points._instance = None
    Points._initialized = False
    Points._points = []
    Points._index_next_point = 1

def test_points_singleton():
    pts1 = Points(2)
    with pytest.raises(ValueError, match="The Points object must be single per analysis."):
        pts2 = Points(2)

def test_points_add_and_retrieve():
    pts = Points(2)
    p1 = Point([0.0, 0.0])
    p2 = Point([1.0, 1.0])
    
    # Add single point via call
    pts(p1)
    # Add list of points via add
    pts.add([p2])
    
    assert pts.number_of_points == 2
    
    # Retrieve via call (nodal indexation is 1-based for the user call)
    assert pts(1) == p1
    assert pts(2) == p2
    
    # Retrieve via point()
    assert pts.point(1) == p1

def test_points_calculating_dof_numbers():
    pts = Points(2)
    
    p1 = Point([0.0, 0.0])
    pts(p1)
    
    # Check automatically calculated dofs
    assert p1.dof_numbers == [0, 1]
    
    p2 = Point([1.0, 1.0])
    pts(p2)
    assert p2.dof_numbers == [2, 3]

def test_points_str_print(capsys):
    pts = Points(2)
    assert str(pts) == "Points()"
    
    p = Point([0.0, 0.0])
    pts(p)
    
    assert "Point[1]" in str(pts)
    
    pts.print()
    captured = capsys.readouterr()
    assert "Point[1]" in captured.out
