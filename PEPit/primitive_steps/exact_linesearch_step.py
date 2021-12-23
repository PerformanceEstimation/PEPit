from PEPit.point import Point


def exact_linesearch_step(x0, f, directions):
    """
    This routines outputs some :math:`x` by *mimicking* an exact line/span search in specified directions.
    It is used for instance in ``PEPit.examples.unconstrained_convex_minimization.wc_gradient_exact_line_search``
    and in ``PEPit.examples.unconstrained_convex_minimization.wc_conjugate_gradient``.

    The routine aims at mimicking the operation:

    .. math::
        :nowrap:

        \\begin{eqnarray}
            x & = & x_0 - \\sum_{i=1}^{T} \\gamma_i d_i,\\\\
            \\text{with } \\overrightarrow{\\gamma} & = & \\arg\\min_\\overrightarrow{\\gamma} f\\left(x_0 - \\sum_{i=1}^{T} \\gamma_i d_i\\right),
        \\end{eqnarray}

    where :math:`T` denotes the number of directions :math:`d_i`. This operation can equivalently be described
    in terms of the following conditions:

    .. math::
        :nowrap:

        \\begin{eqnarray}
            x - x_0 & \\in & \\text{span}\\left\{d_1,\\ldots,d_T\\right\}, \\\\
            \\nabla f(x) & \\perp & \\text{span}\\left\{d_1,\\ldots,d_T\\right\}.
        \\end{eqnarray}

    In this routine, we instead constrain :math:`x_{t}` and :math:`\\nabla f(x_{t})` to satisfy

    .. math::
        :nowrap:

        \\begin{eqnarray}
            \\forall i=1,\\ldots,T: & \\left< \\nabla f(x);\, d_i \\right>  & = & 0,\\\\
            \\text{and } & \\left< \\nabla f(x);\, x - x_0 \\right> & = & 0,
        \\end{eqnarray}

    which is a relaxation of the true line/span search conditions.

    Note:
        The latest condition is automatically implied by the 2 previous ones.

    Warning:
        One can notice this routine does not encode completely the fact that
        :math:`x_{t+1} - x_t` must be a linear combination of the provided directions
        (i.e., this routine performs a relaxation). Therefore, if this routine is included in a PEP,
        the obtained value might be an upper bound on the true worst-case value.

        Although not always tight, this relaxation is often observed to deliver pretty accurate results
        (in particular, it automatically produces tight results under some specific conditions, see, e.g., [1]).
        Two such examples are provided in the `conjugate gradient` and `gradient with exact line search` example files.

    References:
        `[1] Y. Drori and A. Taylor (2020). Efficient first-order methods for convex minimization: a constructive approach.
        Mathematical Programming 184 (1), 183-220.
        <https://arxiv.org/pdf/1803.05676.pdf>`_

    Args:
        x0 (Point): the starting point.
        f (Function): the function on which the (sub)gradient will be evaluated.
        directions (List of Points): the list of all directions required to be orthogonal to the (sub)gradient of x.

    Returns:
        x (Point): such that all vectors in directions are orthogonal to the (sub)gradient of f at x.
        gx (Point): a (sub)gradient of f at x.
        fx (Expression): the function f evaluated at x.

    """

    # Instantiate a Point
    x = Point()

    # Define gradient and function value of f on x
    gx, fx = f.oracle(x)

    # Add constraints
    f.add_constraint((x - x0) * gx == 0)
    for d in directions:
        f.add_constraint(d * gx == 0)

    # Return triplet of points
    return x, gx, fx
