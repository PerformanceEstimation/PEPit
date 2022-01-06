from PEPit.point import Point


def inexact_gradient_step(x0, f, gamma, epsilon, notion='absolute'):
    """
    This routines performs a step :math:`x \\leftarrow x_0 - \\gamma d_{x_0}`
    where :math:`d_{x_0}` is close to the gradient of :math:`f` in :math:`x_0`
    in the following sense:

    .. math:: \\|d_{x_0} - \\nabla f(x_0)\\|^2 \\leqslant \\left\{
              \\begin{eqnarray}
                  & \\varepsilon^2                        & \\text{if notion is set to 'absolute'}, \\\\
                  & \\varepsilon^2 \\|\\nabla f(x_0)\\|^2 & \\text{if notion is set to 'relative'}.
              \\end{eqnarray}
              \\right.

    This relative approximation is used at least in in 3 PEPit examples,
    in particular in 2 unconstrained convex minimizations:
    an inexact gradient descent, and an inexact accelerated gradient.

    Args:
        x0 (Point): starting point x0.
        f (Function): a function.
        gamma (float): the step size parameter.
        epsilon (float): the required accuracy.
        notion (string): defines the mode (absolute or relative inaccuracy).

    Returns:
        x (Point): the output point.
        dx0 (Point): the approximate (sub)gradient of f at x0.
        fx0 (Expression): the value of the function f at x0.

    Raises:
        ValueError: if notion is not set in ['absolute', 'relative'].

    Note:
        When :math:`\\gamma` is set to 0, then the this routine returns
        :math:`x_0`, :math:`d_{x_0}`, and :math:`f_{x_0}`.
        It is used as is in the example of unconstrained convex minimization scheme called
        "inexact gradient exact line search" only to access to the direction :math:`d_{x_0}`
        close to the gradient :math:`g_{x_0}`.

    """

    # Get the gradient gx0 and function value fx0 of f in x0.
    gx0, fx0 = f.oracle(x0)

    # Define dx0 as a proxy to gx0.
    dx0 = Point()
    if notion == 'absolute':
        f.add_constraint((gx0 - dx0) ** 2 - epsilon ** 2 <= 0)
    elif notion == 'relative':
        f.add_constraint((gx0 - dx0) ** 2 - epsilon ** 2 * (gx0 ** 2) <= 0)
    else:
        raise ValueError("inexact_gradient_step supports only notion in ['absolute', 'relative'],"
                         " got {}".format(notion))

    # Perform an inexact gradient step in the direction dx0.
    x = x0 - gamma * dx0

    # Return the newly obtained point, the direction of descent and the value of f in x0.
    return x, dx0, fx0
