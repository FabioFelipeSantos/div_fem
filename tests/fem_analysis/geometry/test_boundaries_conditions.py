import pytest
from div_fem.fem_analysis.geometry.point import Point
from div_fem.fem_analysis.geometry.boundary_condition import BoundaryCondition
from div_fem.fem_analysis.geometry.boundaries_conditions import BoundariesConditions

@pytest.fixture(autouse=True)
def reset_singletons():
    BoundariesConditions._instance = None
    BoundariesConditions._initialized = False
    BoundariesConditions.boundaries_cond = []
    BoundariesConditions._index_next_boundary_cond = 1
    yield
    BoundariesConditions._instance = None
    BoundariesConditions._initialized = False
    BoundariesConditions.boundaries_cond = []
    BoundariesConditions._index_next_boundary_cond = 1

def test_boundaries_conditions_singleton():
    bc1 = BoundariesConditions()
    with pytest.raises(ValueError, match="The Boundaries Conditions object must be single per analysis."):
        bc2 = BoundariesConditions()

def test_boundaries_conditions_add_and_retrieve():
    bcs = BoundariesConditions()
    p = Point([0.0, 0.0])
    bc1 = BoundaryCondition(p, {"x": 0.0})
    bc2 = BoundaryCondition(p, {"y": 0.0})
    
    # Add via call
    bcs(bc1)
    # Add list
    bcs.add([bc2])
    
    assert bcs.count == 2
    
    # Retrieve via call (0-indexed internally)
    assert bcs(0) == bc1
    assert bcs(1) == bc2
    
def test_boundaries_conditions_str():
    bcs = BoundariesConditions()
    p = Point([0.0, 0.0])
    bc1 = BoundaryCondition(p, {"x": 0.0})
    bcs(bc1)
    
    s = str(bcs)
    assert "Conditions(" in s
    assert "1: Condition" in s
