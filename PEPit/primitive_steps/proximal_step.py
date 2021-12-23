from PEPit.point import Point
from PEPit.expression import Expression


def proximal_step(x0, f, gamma):
    """
    This routine performs a proximal step of step-size **gamma**, starting from **x0**, and on function **f**.
    That is, it performs:

    .. math::
        :nowrap:

        \\begin{eqnarray}
            x \\triangleq \\text{prox}_{\\gamma f}(x_0) & \\triangleq & \\arg\\min_x \\left\\{ \\gamma f(x) + \\frac{1}{2} \\|x - x_0\\|^2 \\right\\}, \\\\
            & \\Updownarrow & \\\\
            0 & = & \\gamma g_x + x - x_0 \\text{ for some } g_x\\in\\partial f(x),\\\\
            & \\Updownarrow & \\\\
            x & = & x_0 - \\gamma g_x \\text{ for some } g_x\\in\\partial f(x).
        \\end{eqnarray}

    Args:
        x0 (Point): starting point x0.
        f (Function): function on which the proximal step is computed.
        gamma (float): step-size of the proximal step.

    Returns:
        x (Point): proximal point.
        gx (Point): the (sub)gradient of f at x.
        fx (Expression): the function value of f on x.

    """

    # Define gradient and function value on x.
    gx = Point()
    fx = Expression()

    # Compute x from the docstring equation.
    x = x0 - gamma * gx

    # Add point to Function f.
    f.add_point((x, gx, fx))

    return x, gx, fx
