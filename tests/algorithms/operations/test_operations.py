import pytest
from div_fem.matrices.base_matrix import Matrix
from div_fem.matrices.base_vector import Vector
from div_fem.algorithms.operations.array_scalar_multiplication import array_scalar_multiplication
from div_fem.algorithms.operations.matrix_multiplication import matrix_multiplication
from div_fem.algorithms.operations.matrix_vector_multiplication import matrix_vector_multiplication
from div_fem.algorithms.operations.solve_system import solve_lin_system

def test_array_scalar_multiplication():
    m = Matrix([[1, 2], [3, 4]])
    v = Vector([1, 2])
    
    # Test Matrix
    # scalar = 0
    res_m0 = array_scalar_multiplication(m, 0.0)
    assert res_m0.get_list() == [[0, 0], [0, 0]]
    
    # scalar = 1
    res_m1 = array_scalar_multiplication(m, 1.0)
    assert res_m1.get_list() == [[1, 2], [3, 4]]
    
    # scalar = 2
    res_m2 = array_scalar_multiplication(m, 2.0)
    assert res_m2.get_list() == [[2, 4], [6, 8]]
    
    # Test Vector
    # scalar = 0
    res_v0 = array_scalar_multiplication(v, 0.0)
    assert res_v0.get_list() == [0, 0]
    
    # scalar = 1
    res_v1 = array_scalar_multiplication(v, 1.0)
    assert res_v1.get_list() == [1, 2]
    
    # scalar = 2
    res_v2 = array_scalar_multiplication(v, 2.0)
    assert res_v2.get_list() == [2, 4]

def test_matrix_multiplication():
    m1 = Matrix([[1, 0], [0, 2]])
    m2 = Matrix([[3, 0], [0, 4]])
    res = matrix_multiplication(m1, m2)
    assert res.get_list() == [[3, 0], [0, 8]]
    
    # Error case
    m_bad = Matrix([[1]])
    with pytest.raises(ValueError, match="To multiply two matrix the columns of the first must be the same of rows of the second. Received 2 as matrix A columns and 1 as matrix B rows"):
        matrix_multiplication(m1, m_bad)

def test_matrix_vector_multiplication():
    m = Matrix([[1, 0], [0, 2]])
    v = Vector([3, 4])
    res = matrix_vector_multiplication(m, v)
    assert isinstance(res, Vector)
    assert res.get_list() == [3, 8]
    
    # Error case
    v_bad = Vector([1])
    with pytest.raises(ValueError, match="To multiply a matrix and a vector the columns of the matrix must be the same of rows of the vector. Received 2 as matrix columns and 1 as vector rows"):
        matrix_vector_multiplication(m, v_bad)

def test_solve_lin_system():
    # 2x1 + 1x2 = 5
    # 1x1 + 3x2 = 10
    # True solution: x1=1, x2=3
    coeff = Matrix([[2.0, 1.0], [1.0, 3.0]])
    rhs = Vector([5.0, 10.0])
    
    solution = solve_lin_system(coeff, rhs)
    assert isinstance(solution, Vector)
    assert pytest.approx(solution[0]) == 1.0
    assert pytest.approx(solution[1]) == 3.0

def test_solve_lin_system_with_lists_input():
    # 2x1 + 1x2 = 5
    # 1x1 + 3x2 = 10
    # True solution: x1=1, x2=3
    coeff = [[2.0, 1.0], [1.0, 3.0]]
    rhs = [5.0, 10.0]
    
    solution = solve_lin_system(coeff, rhs)
    assert isinstance(solution, Vector)
    assert pytest.approx(solution[0]) == 1.0
    assert pytest.approx(solution[1]) == 3.0


from div_fem.algorithms.operations.lu_decomposition import LU_decomposition
from div_fem.algorithms.operations.forward_elimination import forward_elimination
from div_fem.algorithms.operations.back_substitution import back_substitution


def test_lu_decomposition_details():
    # Test alternative input and >= 3x3 for complete coverage
    m = [[2, -1, -2],
         [-4, 6, 3],
         [-4, -2, 8]]
    L, U = LU_decomposition(m)
    assert isinstance(L, Matrix)
    assert isinstance(U, Matrix)
    
    # Check if LU == m
    m_reconstructed = matrix_multiplication(L, U)
    assert pytest.approx(m_reconstructed[0, 0]) == 2.0
    assert pytest.approx(m_reconstructed[1, 1]) == 6.0
    assert pytest.approx(m_reconstructed[2, 2]) == 8.0

    # Test error non-square
    with pytest.raises(ValueError, match="A LU decomposition is only available for square matrices."):
        LU_decomposition([[1, 2, 3], [4, 5, 6]])


def test_forward_elimination_details():
    # Test alternative inputs (lists)
    L = [[1, 0], [2, 1]]
    rhs = [5, 10]
    
    sol = forward_elimination(L, rhs)
    assert isinstance(sol, Vector)
    assert pytest.approx(sol[0]) == 5.0
    assert pytest.approx(sol[1]) == 0.0
    
    # Test error dimension mismatch
    with pytest.raises(ValueError, match="An incorrect number of dimensions in the system was found."):
        forward_elimination([[1, 0], [2, 1]], [1, 2, 3])


def test_back_substitution_details():
    # Test alternative inputs (lists)
    U = [[2, 1], [0, 3]]
    rhs = [5, 6]
    
    sol = back_substitution(U, rhs)
    assert isinstance(sol, Vector)
    assert pytest.approx(sol[1]) == 2.0
    assert pytest.approx(sol[0]) == 1.5
    
    # Test error dimension mismatch
    with pytest.raises(ValueError, match="An incorrect number of dimensions in the system was found."):
        back_substitution([[2, 1], [0, 3]], [1, 2, 3])
