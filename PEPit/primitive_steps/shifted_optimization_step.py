from PEPit.point import Point
from PEPit.expression import Expression


def shifted_optimization_step(dir, f):
    """
    This routine outputs a stationary point of a minimization problem:

    .. math:: \\arg\\min_{x} f(x)-\\left< \\text{dir};\, x \\right>.

    That is, it outputs :math:`x` such that

    .. math:: \\text{dir} \\in \\partial f(x).

    Args:
        dir (Point): direction of optimization
        f (Function): function
        
    Returns:
        x (Point): the optimal point.
        gx (Point): the (sub)gradient of f at x, i.e. `dir`.
        fx (Expression): the function value of f at x.

    """

    # Define triplet x, gradient, function value.
    x = Point()
    gx = dir
    fx = Expression()

    # Store it in f list of points.
    f.add_point((x, gx, fx))

    # Return triplet x, gradient, function value.
    return x, gx, fx
