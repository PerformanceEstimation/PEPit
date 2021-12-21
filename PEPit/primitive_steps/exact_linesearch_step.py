from PEPit.point import Point


def exact_linesearch_step(x0, f, directions):
    """
    This routines *mimics* an exact line search in specified directions.

    At each iteration :math:`t`:

    .. math::
        :nowrap:

        \\begin{eqnarray}
            x_{t+1} & = & x_t - \\sum_i \\gamma_t^{(i)} d_t^{(i)} \\\\
            \\text{with } \\overrightarrow{\\gamma_t} & = & \\arg\\min_\\overrightarrow{\\gamma} f\\left(x_t - \\sum_i \\gamma^{(i)} d_t^{(i)}\\right)
        \\end{eqnarray}

    This is used for instance in
    ``PEPit.examples.unconstrained_convex_minimization.wc_gradient_exact_line_search``
    in the last gradient direction:

    .. math::
        :nowrap:

        \\begin{eqnarray}
            x_{t+1} & = & x_t - \\gamma_t \\nabla f(x_t) \\\\
            \\text{with } \\gamma_t & = & \\arg\\min_\\gamma f\\left(x_t - \\gamma \\nabla f(x_t)\\right)
        \\end{eqnarray}

    and in
    ``PEPit.examples.unconstrained_convex_minimization.wc_conjugate_gradient``
    in all the previous directions:

    .. math::
        :nowrap:

        \\begin{eqnarray}
            x_{t+1} & = & x_t - \\sum_{i=0}^{t} \\gamma_t^{(i)} \\nabla f(x_i) \\\\
            \\text{with } \\overrightarrow{\\gamma_t} & = & \\arg\\min_\\overrightarrow{\\gamma} f\\left(x_t - \\sum_{i=0}^{t} \\gamma^{(i)} \\nabla f(x_i)\\right)
        \\end{eqnarray}

    This iteration is then characterized by the 2 following conditions:

    .. math::
        :nowrap:

        \\begin{eqnarray}
            x_{t+1} - x_t & \\in & \\text{span}\\left\{\\left(d_t^{(i)}\\right)_i\\right\} \\\\
            \\nabla f(x_{t+1}) & \\in & \\text{span}\\left\{\\left(d_t^{(i)}\\right)_i\\right\}^T
        \\end{eqnarray}

    In this routine, we define each new triplet :math:`\\left(x_{t+1}, \\nabla f(x_{t+1}), f(x_{t+1})\\right)` such that

    .. math::
        :nowrap:

        \\begin{eqnarray}
            \\forall i, & \\left< \\nabla f(x_{t+1}) \\Big| d_t^{(i)} \\right>  & = & 0 \\\\
            \\text{and } & \\left< \\nabla f(x_{t+1}) \\Big| x_{t+1} - x_t \\right> & = & 0
        \\end{eqnarray}

    Note:
        The latest condition is automatically implied by the 2 previous ones.

    Warning:
        One can notice this routine does not encode completely the fact that
        :math:`x_{t+1} - x_t` must be a linear combination of the provided directions.
        Hence, if this routine is included in one PEP,
        the obtained value is an upper bound of the real worst-case value.

        On the other hand,
        it can be shown that the optimum of commons PEPs using this routine
        verifies the aforementioned implicit constraint.

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
