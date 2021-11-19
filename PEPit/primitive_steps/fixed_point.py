from PEPit.point import Point
from PEPit.expression import Expression


def fixed_point(A):
    """
    This routine allows to find a fixed point of an operator A, that is x suh that:
    Ax = x.

    Args:
        A (Function): an operator or a function

    Returns:
         - if A is an operator, x such that x = A x (fixed point of A)
         - if A is a function, x such that x = A.gradient(x)

    """

    x = Point()
    fx = Expression()
    A.add_point((x, x, fx))
    return x, x, fx
