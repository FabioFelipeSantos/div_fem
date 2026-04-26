import pytest
import numpy as np
from div_fem.matrices.base_vector import Vector
from div_fem.matrices.base_matrix import Matrix
from div_fem.fem_analysis.geometry.point import Point


def test_vector_init_with_elements():
    v = Vector([1, 2, 3])
    assert v.rows == 3
    assert v.get_list() == [1, 2, 3]

    p = Point([4, 5])
    v2 = Vector(p)
    assert v2.rows == 2
    assert v2.get_list() == [4, 5]


def test_vector_init_with_rows():
    v = Vector(rows=4)
    assert v.rows == 4
    assert v.get_list() == [0, 0, 0, 0]


def test_vector_init_exceptions():
    with pytest.raises(ValueError):
        Vector()

    with pytest.raises(ValueError):
        Vector([1, 2], rows=2)

    with pytest.raises(ValueError):
        Vector(rows=2, random=True, unity_direction_vector_dim=1)


def test_vector_shape():
    v = Vector([1, 2, 3])
    assert v.shape == (3, 1)


def test_vector_transpose():
    v = Vector([1, 2, 3])
    T = v.T
    assert isinstance(T, Matrix)
    assert T.shape == (1, 3)
    assert T[0, 0] == 1
    assert T[0, 1] == 2
    assert T[0, 2] == 3


def test_vector_norm():
    v = Vector([3, 4])
    assert pytest.approx(v.norm) == 5.0


def test_vector_enumeration():
    v = Vector([10, 20])
    enum_list = list(v.enumeration)
    assert enum_list == [(0, 10), (1, 20)]


def test_vector_index():
    v = Vector([10, 20, 30])
    assert v.index(20) == 1


def test_vector_dot():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    assert v1.dot(v2) == 32
    assert v1.dot([4, 5, 6]) == 32


def test_vector_add():
    v1 = Vector([1, 2])
    v2 = Vector([3, 4])
    res = v1 + v2
    assert res.get_list() == [4.0, 6.0]
    
    res2 = v1 + [1, 1]
    assert res2.get_list() == [2.0, 3.0]


def test_vector_iadd():
    v1 = Vector([1, 2])
    v1 += Vector([3, 4])
    assert v1.get_list() == [4, 6]


def test_vector_sub():
    v1 = Vector([5, 5])
    v2 = Vector([2, 3])
    res = v1 - v2
    assert res.get_list() == [3.0, 2.0]


def test_vector_len():
    v = Vector([1, 2, 3, 4])
    assert len(v) == 4


def test_vector_mul():
    v = Vector([1, 2])
    res = v * 3
    assert res.get_list() == [3.0, 6.0]

    # Matrix multiplication
    m = Matrix([[1, 2]])
    res2 = v * m
    assert res2.shape == (2, 2)
    assert res2[0, 0] == 1
    assert res2[0, 1] == 2
    assert res2[1, 0] == 2
    assert res2[1, 1] == 4


def test_vector_rmul():
    v = Vector([1, 2])
    res = 3 * v
    assert res.get_list() == [3.0, 6.0]


def test_vector_getitem():
    v = Vector([10, 20, 30])
    assert v[1] == 20
    sub_v = v[[0, 2]]
    assert isinstance(sub_v, Vector)
    assert sub_v.get_list() == [10, 30]


def test_vector_setitem():
    v = Vector([10, 20, 30])
    v[1] = 25
    assert v.get_list() == [10, 25, 30]

    v[[0, 2]] = Vector([15, 35])
    assert v.get_list() == [15, 25, 35]

    v[[0, 1]] = [1, 2]
    assert v.get_list() == [1, 2, 35]


def test_vector_statics():
    vz = Vector.zeros(3)
    assert vz.get_list() == [0, 0, 0]

    vo = Vector.ones(2)
    assert vo.get_list() == [1, 1]

    vr = Vector.random(2)
    assert len(vr.get_list()) == 2

    vu = Vector.unit_direction_vector(3, 1)
    assert vu.get_list() == [0, 1, 0]

    norm = Vector.vector_norm([3, 4])
    assert norm == 5.0

    dot = Vector.vecdot([1, 2], [3, 4])
    assert dot == 11.0


def test_vector_init_unity_direction():
    vu = Vector(rows=3, unity_direction_vector_dim=1)
    assert vu.get_list() == [0, 1, 0]


def test_vector_print_specifiers(capsys):
    v = Vector([1.234567, 100])
    
    v.type_of_print_specifier = "decimal"
    v.print()
    captured = capsys.readouterr()
    assert "1.2346" in captured.out
    
    v.type_of_print_specifier = "scientific"
    v.print()
    captured = capsys.readouterr()
    assert "1.234567E+00" in captured.out
    
    v.type_of_print_specifier = "integer"
    v.print()
    captured = capsys.readouterr()
    assert "   1" in captured.out


def test_vector_repr():
    v = Vector([1, 2])
    rep = repr(v)
    assert rep.startswith("Vector(")
    assert rep.endswith(")")


def test_vector_iadd_exceptions():
    v = Vector([1, 2])
    v += [3, 4]
    assert v.get_list() == [4, 6]
    
    with pytest.raises(ValueError, match="To sum a vector in place, the second vector must have the same number of rows that the first."):
        v += Vector([1, 2, 3])


def test_vector_mul_exceptions():
    v = Vector([1, 2])
    
    res = v * [[1, 2, 3]]
    assert isinstance(res, Matrix)
    assert res.shape == (2, 3)
    
    with pytest.raises(ValueError, match="Shape mismatch error for matrix. Hope to get 1, received 2."):
        v * Matrix([[1, 2], [3, 4]])


def test_vector_setitem_exceptions():
    v = Vector([10, 20])
    
    with pytest.raises(ValueError, match="Single element assignment requires a scalar value."):
        v[0] = [1]
        
    with pytest.raises(ValueError, match="The value for a list of indexes must be a valid Vector or list of scalars with the same size of the indexes"):
        v[[0]] = 1
        
    with pytest.raises(ValueError, match="Shape mismatch: expected 1 rows, got 2"):
        v[[0]] = [1, 2]


def test_vector_statics_norm_vecdot():
    v = Vector([3, 4])
    assert Vector.vector_norm(v) == 5.0
    
    with pytest.raises(ValueError, match="Shape mismatch error. For the dot product must vectors must be the same size."):
        Vector.vecdot([1, 2], [1, 2, 3])
