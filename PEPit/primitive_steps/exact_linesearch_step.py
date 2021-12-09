from PEPit.point import Point


def exact_linesearch_step(x0, f, directions):
    """
    This routines *mimics* an exact line search.
    This is used for instance in

    PEPit.examples.a_methods_for_unconstrained_convex_minimization.gradient_exact_line_search.py

    At each iteration k:

    .. math::

        gamma = \arg\min_t f\\left(x_k - tf'(x_k)\\right)
        x_{k+1} = x_k - \\gamma * f(x_k)

    The two iterated are thus defined by the following two conditions :

    .. math::

        x_{k+1} - x_{k} + \\gamma f'(x_k) = 0
        f'(x_{k+1})^T(x_{k+1} - x_k) = 0

    In this routine, we define each new triplet :math:`\\left(x_{k+1}, f'(x_{k+1}), f(x_{k+1})\\right)` such that

    .. math::

        f'(x_{k+1})^Tf'(x_k) = 0

    Args:
        x0 (Point): the starting point.
        f (Function): the function on which the (sub)gradient will be evaluated.
        directions (List of Points): the list of all directions required to be orthogonal to the (sub)gradient of x.
                                     Note that (x-x0) is automatically constrained to be orthogonal
                                     to the subgradient of f at x.

    Returns:
        x (Point), such that all vectors in directions are orthogonal to the (sub)gradient of f at x.
        gx (Point), a (sub)gradient of f at x.
        fx (Expression), the function f evaluated at x.

    """

    x = Point()
    gx, fx = f.oracle(x)
    f.add_constraint((x - x0) * gx == 0)
    for d in directions:
        f.add_constraint(d * gx == 0)

    return x, gx, fx
