from PEPit.point import Point
from PEPit.expression import Expression


def shifted_optimization_step(dir, f):
    """
    This routine outputs a stationary point of a minimization problem:

    .. math:: \\arg\\min_{x} f(x)-\\left< \\text{dir};\, x \\right>.

    That is, it outputs :math:`x` such that

    .. math:: \\text{dir} \\in \\partial f(x).

    Shifted optimization oracles are classically used in difference-of-convex algorithms
    (a.k.a., convex-concave procedure), see, e.g., [1].

    References:
    	`[1] H.A. Le Thi, T. Pham Dinh (2018).
    	DC programming and DCA: thirty years of developments.
    	Mathematical Programming, 169(1), 5-68.
    	<https://link.springer.com/article/10.1007/s10107-018-1235-y>`_


    Args:
        dir (Point): direction/linear shift in the objective of the optimization problem
        f (Function): function
        
    Returns:
        x (Point): oracle output.
        gx (Point): the (sub)gradient of f at x.
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
