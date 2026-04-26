import pytest
import math
from div_fem.algorithms.integration.gauss_quadrature import gauss_quadrature
from div_fem.matrices.base_vector import Vector

def exact_integral_x_power(p: int) -> float:
    """Exact integral of x^p from -1 to 1"""
    if p % 2 != 0:
        return 0.0
    return 2.0 / (p + 1)

def test_gauss_quadrature_value_error():
    def dummy_func(x: float) -> Vector:
        return Vector([x])

    with pytest.raises(ValueError, match="The Gauss Quadrature number of points must be one of:"):
        gauss_quadrature(dummy_func, 7)
        
    with pytest.raises(ValueError, match="The Gauss Quadrature number of points must be one of:"):
        gauss_quadrature(dummy_func, 0)

def test_gauss_quadrature_polynomials():
    """
    Test polynomials up to degree 2n - 1 for n=1..6
    """
    for n in range(1, 7):
        max_degree = 2 * n - 1
        
        for p in range(max_degree + 1):
            # Default argument binds p correctly in the lambda/closure
            def poly_func(x: float, power=p) -> Vector:
                return Vector([x ** power])
            
            res = gauss_quadrature(poly_func, n)
            exact = exact_integral_x_power(p)
            assert pytest.approx(res[0], rel=1e-9, abs=1e-9) == exact

def test_gauss_quadrature_erroneous_integration():
    """
    An n-point quadrature cannot exactly integrate a polynomial of degree 2n.
    For n=2, max exact degree is 3. Let's test degree 4.
    Exact integral of x^4 from -1 to 1 is 2/5 = 0.4
    """
    def degree_4(x: float) -> Vector:
        return Vector([x ** 4])
    
    res_2_points = gauss_quadrature(degree_4, 2)
    exact = 0.4
    
    # 2 points should fail to be exact, creating an "erroneous" integration
    assert res_2_points[0] != pytest.approx(exact, rel=1e-9, abs=1e-9)
    
    # But 3 points should be exact since 2(3)-1 = 5 >= 4
    res_3_points = gauss_quadrature(degree_4, 3)
    assert pytest.approx(res_3_points[0], rel=1e-9, abs=1e-9) == exact

def test_gauss_quadrature_transcendental():
    """
    Integrate transcendental function e^x from -1 to 1.
    Exact is e - 1/e
    """
    def exp_func(x: float) -> Vector:
        return Vector([math.exp(x)])
    
    res = gauss_quadrature(exp_func, 6)
    exact = math.exp(1) - math.exp(-1)
    
    # 6 points Gauss Quadrature should be extremely accurate for e^x
    assert pytest.approx(res[0], rel=1e-9, abs=1e-9) == exact
