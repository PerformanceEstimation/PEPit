from PEPit.point import Point
from PEPit.expression import Expression


def linear_optimization_step(dir, ind):
    """
    This routine outputs the result of a minimization problem with linear objective (whose direction
    is provided by `dir`) on the domain of the (closed convex) indicator function `ind`.
    That is, it outputs a solution to

    .. math:: \\arg\\min_{\\text{ind}(x)=0} \\left< \\text{dir};\, x \\right>,

    One can notice that :math:`x` is solution of this problem if and only if

    .. math:: - \\text{dir} \\in \\partial \\text{ind}(x).

    Args:
        dir (Point): direction of optimization
        ind (ConvexIndicatorFunction): convex indicator function

    Returns:
        x (Point): the optimal point.
        gx (Point): the (sub)gradient of ind on x.
        fx (Expression): the function value of ind on x.

    """

    # Define triplet x, gradient, function value.
    x = Point()
    gx = - dir
    fx = Expression()

    # Store it in ind list of points.
    ind.add_point((x, gx, fx))

    # Return triplet x, gradient, function value.
    return x, gx, fx
