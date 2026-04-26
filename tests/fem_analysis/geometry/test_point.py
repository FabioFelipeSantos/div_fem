import pytest
from div_fem.fem_analysis.geometry.point import Point

def test_point_init():
    p = Point([1.0, 2.0, 3.0])
    assert p.dimension == 3
    assert p.get_list() == [1.0, 2.0, 3.0]
    assert len(p) == 3

def test_point_norm():
    p = Point([3.0, 4.0])
    assert pytest.approx(p.norm) == 5.0

def test_point_str_repr():
    p = Point([1.0, -2.5])
    assert str(p) == "( 1.00, -2.50)"
    assert repr(p) == "Point( 1.00, -2.50)"

def test_point_print(capsys):
    p = Point([1.0, -2.5])
    p.print()
    captured = capsys.readouterr()
    assert "Point( 1.00, -2.50)" in captured.out

def test_point_descriptors_exceptions():
    p = Point([0.0, 0.0])
    
    with pytest.raises(AttributeError, match="The Point index can only be accessed after it being in Points class"):
        _ = p.index
        
    with pytest.raises(AttributeError, match="The DOFNumbers can only be accessed after the point being in the Point class with an Structural Analysis class instantiated"):
        _ = p.dof_numbers
        
    with pytest.raises(ValueError, match="A index for point must be greater than 0."):
        p.index = -1

def test_point_getitem():
    p = Point([10.0, 20.0])
    assert p[0] == 10.0
    assert p[1] == 20.0
    
    with pytest.raises(IndexError, match="Received 2 as index for a point with 2 coordinates."):
        _ = p[2]

def test_point_setitem():
    p = Point([10.0, 20.0])
    
    # Valid scalar assign
    p[0] = 15.0
    assert p[0] == 15.0
    
    # Valid list assign
    p[[0, 1]] = [100.0, 200.0]
    assert p.get_list() == [100.0, 200.0]
    
    # Index > dimension scalar
    with pytest.raises(IndexError, match="The provided index 3 is greater than dimension 2 of the point."):
        p[3] = 1.0
        
    # Value is not float for scalar
    with pytest.raises(ValueError, match="For just one coordinate provide one value for it."):
        p[0] = [1.0]

    # Index list > dimension
    with pytest.raises(ValueError, match="The indexes list has more values than the dimension 2 of the point."):
        p[[0, 1, 2]] = [1.0, 2.0, 3.0]
        
    # Values is not list for list index
    with pytest.raises(ValueError, match="To set more than one coordinate for the point, provide a list of values for each coordinate."):
        p[[0, 1]] = 1.0
        
    # len(values) > len(idx)
    with pytest.raises(ValueError, match="The provided values have more values than the list indexes."):
        p[[0]] = [1.0, 2.0]
