from PEPit.point import Point
from PEPit.expression import Expression


def epsilon_subgradient_step(x0, f, gamma):
    """
    This routines performs a step :math:`x \\leftarrow x_0 - \\gamma d_{x_0}`
    where :math:`d_{x_0} \\in\\partial_{\\varepsilon} f(x_0)`. That is, :math:`d_{x_0}` is an
    :math:`\\varepsilon`-subgradient of :math:`f` at :math:`x_0`. The set :math:`\\partial_{\\varepsilon} f(x_0)`
    (referred to as the :math:`\\varepsilon`-subdifferential) is defined as (see [1, Section 3])

    .. math:: \\partial_{\\varepsilon} f(x)=\\left\\{g:\, f(z)\\geqslant f(x)+\\left< g;\, z-x \\right>-\\varepsilon \\right\\}

    References:
        `[1] A. Brøndsted, R.T. Rockafellar.
        On the subdifferentiability of convex functions.
        Proceedings of the American Mathematical Society 16(4), 605–611 (1965)
        <https://www.jstor.org/stable/2033889>`_

    Args:
        x0 (Point): starting point x0.
        f (Function): a function.
        gamma (float): the step size parameter.

    Returns:
        x (Point): the output point.
        dx0 (Point): an :math:`\\varepsilon`-subgradient of f at x0.
        fx0 (Expression): the value of the function f at x0.
        epsilon (Expression): the value of epsilon.

    """

    dx0 = Point()
    epsilon = Expression()
    fx0 = Expression()
    
    f.add_point((x0, dx0, fx0-epsilon))
    x = x0 - gamma * dx0
    # Return the newly obtained point, the epsilon-subgradient, the value of f in x0, and epsilon.
    
    return x, dx0, fx0, epsilon
