from typing import Literal, Protocol, Any, TypeVar

from div_fem.matrices.base_matrix import Matrix
from div_fem.matrices.base_vector import Vector

QuadraturePoints = Literal[1, 2, 3, 4, 5]

# WEIGHTS_ABSCISSAE: { n: ([weights, w_i], [abscissa, x_i])}
WEIGHTS_ABSCISSAE: dict[QuadraturePoints, tuple[list[float], list[float]]] = {
    1: ([2], [0]),
    2: ([1, 1], [-0.5773502691896257, 0.5773502691896257]),
    3: (
        [0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
        [-0.7745966692414834, 0, 0.7745966692414834],
    ),
    4: (
        [
            0.3478548451374538,
            0.6521451548625461,
            0.6521451548625461,
            0.3478548451374538,
        ],
        [
            -0.8611363115940526,
            -0.3399810435848563,
            0.3399810435848563,
            0.8611363115940526,
        ],
    ),
    5: (
        [
            0.2369268850561891,
            0.4786286704993665,
            0.5688888888888889,
            0.4786286704993665,
            0.2369268850561891,
        ],
        [
            -0.9061798459386640,
            -0.5384693101056831,
            0.0000000000000000,
            0.5384693101056831,
            0.9061798459386640,
        ],
    ),
}

T = TypeVar("T", Matrix, Vector, covariant=True)


class FunctionTypeWithOptionalKwargs(Protocol[T]):
    def __call__(self, independent_value: float, **kwds: Any) -> T: ...


# Exact for polynomials up to degree of 2n - 1
def gauss_quadrature(
    function: FunctionTypeWithOptionalKwargs[T],
    quadrature_points: QuadraturePoints,
    **kwargs: Any,
) -> T:
    weights, x_values = WEIGHTS_ABSCISSAE[quadrature_points]

    integration_result = function(x_values[0], **kwargs) * weights[0]

    for w, x in zip(weights[1:], x_values[1:]):
        integration_result += function(x, **kwargs) * w

    return integration_result
