from PEPit.point import Point
from PEPit.expression import Expression


def fixed_point(A):
    """
    This routine allows to find a fixed point of the operator :math:`A`, that is :math:`x` such that :math:`Ax = x`.

    Args:
        A (Function): an operator or a function.

    Returns:
        x (Point): a fixed point of A.
        Ax (Point): Ax = x.
        fx (Expression): a function value (useful only if A is a function).

    """

    # Define a point and function value
    x = Point()
    fx = Expression()

    # Add triplet to A list of points (by definition Ax = x)
    A.add_point((x, x, fx))

    # Return the aforementioned triplet
    return x, x, fx
