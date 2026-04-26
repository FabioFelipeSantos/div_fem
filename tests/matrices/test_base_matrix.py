import pytest
from div_fem.matrices.base_matrix import Matrix
from div_fem.matrices.base_vector import Vector
import numpy as np


def test_matrix_init_with_elements():
    m = Matrix([[1, 2], [3, 4]])
    assert m.rows == 2
    assert m.columns == 2
    assert m.get_list() == [[1, 2], [3, 4]]

    with pytest.raises(ValueError):
        Matrix([[1, 2], [3]])


def test_matrix_init_with_shape():
    m = Matrix(rows=2, columns=3)
    assert m.shape == (2, 3)
    assert m.get_list() == [[0, 0, 0], [0, 0, 0]]

    m_sq = Matrix(rows=2)
    assert m_sq.shape == (2, 2)

    m_id = Matrix(rows=2, identity=True)
    assert m_id.get_list() == [[1, 0], [0, 1]]

    with pytest.raises(ValueError):
        Matrix(rows=2, columns=3, identity=True)


def test_matrix_init_exceptions():
    with pytest.raises(ValueError):
        Matrix()

    with pytest.raises(ValueError):
        Matrix([[1]], rows=1)

    with pytest.raises(ValueError):
        Matrix(rows=2, random=True, identity=True)


def test_matrix_transpose():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    T = m.T
    assert T.shape == (3, 2)
    assert T.get_list() == [[1, 4], [2, 5], [3, 6]]


def test_matrix_norm():
    m = Matrix([[3, 0], [0, 4]])
    assert pytest.approx(m.norm) == 5.0


def test_matrix_to_vector():
    m1 = Matrix([[1, 2, 3]])
    v1 = m1.to_vector()
    assert isinstance(v1, Vector)
    assert v1.get_list() == [1, 2, 3]

    m2 = Matrix([[1], [2], [3]])
    v2 = m2.to_vector()
    assert isinstance(v2, Vector)
    assert v2.get_list() == [1, 2, 3]

    with pytest.raises(ValueError):
        Matrix([[1, 2], [3, 4]]).to_vector()


def test_matrix_statics():
    mz = Matrix.zeros(2, 3)
    assert mz.shape == (2, 3)
    assert mz.get_list() == [[0, 0, 0], [0, 0, 0]]

    mo = Matrix.ones(2)
    assert mo.shape == (2, 2)
    assert mo.get_list() == [[1, 1], [1, 1]]

    mr = Matrix.random(2, 2)
    assert mr.shape == (2, 2)


def test_matrix_add():
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[1, 1], [1, 1]])
    res = m1 + m2
    assert res.get_list() == [[2, 3], [4, 5]]


def test_matrix_iadd():
    m1 = Matrix([[1, 2], [3, 4]])
    m1 += Matrix([[1, 1], [1, 1]])
    assert m1.get_list() == [[2, 3], [4, 5]]


def test_matrix_sub():
    m1 = Matrix([[5, 5], [5, 5]])
    m2 = Matrix([[1, 2], [3, 4]])
    res = m1 - m2
    assert res.get_list() == [[4, 3], [2, 1]]


def test_matrix_inv():
    m = Matrix([[1, 2], [3, 4]])
    minv = m.inv()
    assert pytest.approx(minv[0, 0]) == -2.0
    assert pytest.approx(minv[0, 1]) == 1.0
    assert pytest.approx(minv[1, 0]) == 1.5
    assert pytest.approx(minv[1, 1]) == -0.5


def test_matrix_getitem():
    m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert m[1, 2] == 6

    sub_m1 = m[[0, 2]]
    assert isinstance(sub_m1, Matrix)
    assert sub_m1.get_list() == [[1, 3], [7, 9]]

    sub_m2 = m[[0, 1], [1, 2]]
    assert isinstance(sub_m2, Matrix)
    assert sub_m2.get_list() == [[2, 3], [5, 6]]


def test_matrix_setitem():
    m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    m[1, 1] = 100
    assert m[1, 1] == 100

    m[[0, 1], [0, 1]] = [[10, 20], [30, 40]]
    assert m.get_list() == [[10, 20, 3], [30, 40, 6], [7, 8, 9]]

    m[[2], [0, 1, 2]] = [[0, 0, 0]]
    assert m.get_list() == [[10, 20, 3], [30, 40, 6], [0, 0, 0]]


def test_matrix_print(capsys):
    m = Matrix([[1, 2], [3, 4]])
    m.print()
    captured = capsys.readouterr()
    assert "[" in captured.out


def test_matrix_statics_without_n():
    mz = Matrix.zeros(3)
    assert mz.shape == (3, 3)
    mr = Matrix.random(3)
    assert mr.shape == (3, 3)


def test_matrix_repr():
    m = Matrix([[1, 2], [3, 4]])
    rep = repr(m)
    assert rep.startswith("Matrix(")
    assert rep.endswith(")")


def test_matrix_iadd_exceptions():
    m = Matrix([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="To sum a matrix in place, the second matrix must have the same number of rows that the first."):
        m += Matrix([[1, 2]])

    with pytest.raises(ValueError, match="To sum a matrix in place, the second matrix must have the same number of columns that the first."):
        m += Matrix([[1], [2]])

    m += [[1, 2], [3, 4]]
    assert m.get_list() == [[2, 4], [6, 8]]


def test_matrix_mul_details():
    m = Matrix([[1, 2], [3, 4]])
    
    # Int / Float
    res_int = m * 2
    assert res_int.get_list() == [[2, 4], [6, 8]]
    res_float = m * 0.5
    assert res_float.get_list() == [[0.5, 1.0], [1.5, 2.0]]

    # Matrix
    m2 = Matrix([[1, 0], [0, 1]])
    res_m = m * m2
    assert res_m.get_list() == [[1, 2], [3, 4]]

    # Vector
    v = Vector([1, 1])
    res_v = m * v
    assert isinstance(res_v, Vector)
    assert res_v.get_list() == [3, 7]

    # Object with get_list() (not sequence)
    class FakeVector:
        def get_list(self):
            return [1, 1]
    
    res_fake = m * FakeVector()
    assert isinstance(res_fake, Vector)
    assert res_fake.get_list() == [3, 7]

    # 1D Sequence
    res_seq_1d = m * [1, 1]
    assert isinstance(res_seq_1d, Vector)
    assert res_seq_1d.get_list() == [3, 7]

    # 2D Sequence
    res_seq_2d = m * [[1, 0], [0, 1]]
    assert isinstance(res_seq_2d, Matrix)
    assert res_seq_2d.get_list() == [[1, 2], [3, 4]]


def test_matrix_getitem_exceptions():
    m = Matrix([[1, 2], [3, 4]])
    
    with pytest.raises(ValueError, match="This matrix's values can be accessible in the ways"):
        m[[0], 1]
        
    with pytest.raises(ValueError, match="This matrix's values can be accessible in the ways"):
        m[0, [1]]


def test_matrix_setitem_exceptions():
    m = Matrix([[1, 2], [3, 4]])
    
    with pytest.raises(ValueError, match="Single element assignment requires a scalar value."):
        m[0, 0] = [1]
        
    with pytest.raises(ValueError, match="This matrix's values can be accessible in the ways"):
        m[[0], 1] = 1
        
    with pytest.raises(ValueError, match="The value for a list of indexes must be a valid Matrix or list of lists with the same size of the indexes"):
        m[[0], [0]] = 1
        
    with pytest.raises(ValueError, match="Shape mismatch: expected 1 rows, got 2"):
        m[[0], [0]] = [[1], [2]]
        
    with pytest.raises(ValueError, match="Shape mismatch in row 0: expected 1 columns, got 2"):
        m[[0], [0]] = [[1, 2]]
