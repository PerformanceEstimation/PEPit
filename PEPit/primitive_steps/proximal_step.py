from PEPit.point import Point
from PEPit.expression import Expression


def proximal_step(x0, f, step):
    """
    This routine performs a proximal step of step size **step**, starting from **x0**, and on function **f**.
    That is, it performs :

    .. math::
        :nowrap:

        \\begin{eqnarray}
            x \\triangleq \\text{prox}_{\\gamma f}(x_0) & = & \\arg\\min_x { \\gamma f(x) + \\frac{1}{2} \\|x - x_0\\|^2 } \\\\
            & \\Updownarrow & \\\\
            0 & \\in & \\gamma \\partial f(x) + x - x_0 \\\\
            & \\Updownarrow & \\\\
            x & \\in & x_0 - \\gamma \\partial f(x)
        \\end{eqnarray}

    Args:
        x0 (Point): starting point x0.
        f (Function): function on which the proximal step is computed.
        step (float): step size of the proximal step.

    Returns:
        x (Point): proximal point.
        gx (Point): the (sub)gradient of f at x.
        fx (Expression): the function value of f on x.

    """

    # Define gradient and function value on x.
    gx = Point()
    fx = Expression()

    # Compute x from the docstring equation.
    x = x0 - step * gx

    # Add point to Function f.
    f.add_point((x, gx, fx))

    return x, gx, fx
