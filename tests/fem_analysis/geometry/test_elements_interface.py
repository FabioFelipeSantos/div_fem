import pytest
from div_fem.fem_analysis.geometry.elements_interface import ElementInterface, index_validation
from div_fem.fem_analysis.geometry.point import Point

class DummyElement(ElementInterface):
    @property
    def points(self): return list(self.extreme_points)
    @property
    def degree_of_freedom(self): return []
    @property
    def T(self): return None
    @property
    def local_stiffness_matrix(self): return None
    @property
    def local_forces_vector(self): return None
    
    def __init__(self, pts):
        self.extreme_points = pts
        self.number_interpolation_points = len(pts)

def test_index_validation():
    # Deve passar sem exceção
    index_validation(0)
    
    # Deve engatilhar a linha 26
    with pytest.raises(ValueError, match="A index for the element must be greater than 0. Received -1"):
        index_validation(-1)

def test_element_interface_print(capsys):
    from unittest.mock import MagicMock
    
    p1 = MagicMock()
    p1.dof_numbers = [1, 2]
    p1.__repr__ = lambda self: "MockPoint1"
    
    p2 = MagicMock()
    p2.dof_numbers = [3, 4]
    p2.__repr__ = lambda self: "MockPoint2"
    
    elem = DummyElement((p1, p2))
    
    # Linha 84: cobrindo a impressão com argumento idx None ou 0
    elem.print()
    captured = capsys.readouterr()
    assert "MockPoint1" in captured.out
    
    # Linha 86: cobrindo a impressão com argumento idx válido (>0)
    elem.print(idx=1)
    captured = capsys.readouterr()
    assert "MockPoint2" in captured.out

def test_element_interface_str_no_dof_numbers():
    p1 = Point([0.0, 0.0])
    p2 = Point([1.0, 0.0])
    elem = DummyElement((p1, p2))
    
    # Linha 95: quando point não tem dof_numbers definido ou está vazio
    s = str(elem)
    assert "1: " + repr(p1) in s
    assert "DOF" not in s
    
    # Test line 105: __repr__
    r = repr(elem)
    assert repr(p1) in r
    
def test_element_interface_with_dof_and_getitem():
    from unittest.mock import MagicMock
    
    p1 = MagicMock()
    p1.dof_numbers = [1, 2]
    p1.__repr__ = lambda self: "MockPoint1"
    
    p2 = MagicMock()
    p2.dof_numbers = [3, 4]
    p2.__repr__ = lambda self: "MockPoint2"
    
    elem = DummyElement((p1, p2))
    
    # Test line 97: com dof_numbers válidos
    s = str(elem)
    assert "DOF: [1, 2]" in s
    assert "DOF: [3, 4]" in s
    
    # Test lines 114-119: __getitem__
    # Index out of bounds
    with pytest.raises(IndexError, match="The element doesn't have more than 2 points"):
        _ = elem[2]
        
    # Index valid
    item = elem[0]
    assert item[0] == p1
    assert item[1] == [1, 2]
