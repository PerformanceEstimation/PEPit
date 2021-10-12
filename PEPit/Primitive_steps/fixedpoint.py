from PEPit.point import Point
from PEPit.expression import Expression


def fixedpoint(A):
    """
    This tourine allows to find a fixed point of an operator A.

    :param A: an operator or a function
    :return: - if A is an operator, x such that x = A x (ixed point of A)
             - if A is a function, x such that x = A.gradient(x)
    """

    x = Point()
    fx = Expression()
    A.add_point((x, x, fx))
    return x, x, fx
