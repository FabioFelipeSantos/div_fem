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
