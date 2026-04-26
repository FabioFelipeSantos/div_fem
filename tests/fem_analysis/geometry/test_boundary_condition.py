import pytest
from div_fem.fem_analysis.geometry.point import Point
from div_fem.fem_analysis.geometry.boundary_condition import BoundaryCondition

def test_boundary_condition_init_and_props():
    p = Point([0.0, 0.0])
    
    # x fixed, y prescribed, moment free
    bc = BoundaryCondition(p, {"x": 0.0, "y": 5.0, "moment": None}, rotation=45.0)
    
    assert bc.rotation == 45.0
    
    assert bc.is_x_fixed is True
    assert bc.is_y_fixed is False
    assert bc.is_moment_fixed is False
    
    assert bc.prescribed_x == (True, 0.0)
    assert bc.prescribed_y == (True, 5.0)
    assert bc.prescribed_moment == (False, None)
    
def test_boundary_condition_methods():
    p = Point([0.0, 0.0])
    bc = BoundaryCondition(p, {"x": 0.0, "y": 5.0})
    
    # boundary_condition without axis
    d = bc.boundary_condition()
    assert d["x"] == 0.0
    assert d["y"] == 5.0
    assert d["moment"] is None
    
    # boundary_condition with axis
    assert bc.boundary_condition("x") == 0.0
    assert bc.boundary_condition("y") == 5.0
    assert bc.boundary_condition("moment") is None

def test_boundary_condition_str():
    p = Point([0.0, 0.0])
    bc1 = BoundaryCondition(p, {"x": 0.0})
    assert "x: 0.0" in str(bc1)
    
    bc2 = BoundaryCondition(p, {"x": 0.0, "y": 5.0, "moment": 10.0}, rotation=45.0)
    s = str(bc2)
    assert "x: 0.0" in s
    assert "y: 5.0" in s
    assert "moment: 10.0" in s
    assert "rotation: 45.0" in s
