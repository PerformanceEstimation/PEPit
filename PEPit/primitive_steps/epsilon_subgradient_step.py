from PEPit.point import Point
from PEPit.expression import Expression


def epsilon_subgradient_step(x0, f, gamma):
    """
    This routine performs a step :math:`x \\leftarrow x_0 - \\gamma g_0`
    where :math:`g_0 \\in\\partial_{\\varepsilon} f(x_0)`. That is, :math:`g_0` is an
    :math:`\\varepsilon`-subgradient of :math:`f` at :math:`x_0`. The set :math:`\\partial_{\\varepsilon} f(x_0)`
    (referred to as the :math:`\\varepsilon`-subdifferential) is defined as (see [1, Section 3])

    .. math:: \\partial_{\\varepsilon} f(x_0)=\\left\\{g_0:\,\\forall z,\, f(z)\\geqslant f(x_0)+\\left< g_0;\, z-x_0 \\right>-\\varepsilon \\right\\}.
    
    An alternative characterization of :math:`g_0 \\in\\partial_{\\varepsilon} f(x_0)` consists in writing
    
    .. math:: f(x_0)+f^*(g_0)-\\left< g_0;x_0\\right>\\leqslant \\varepsilon.

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
        g0 (Point): an :math:`\\varepsilon`-subgradient of f at x0.
        f0 (Expression): the value of the function f at x0.
        epsilon (Expression): the value of epsilon.

    """

    g0 = Point()
    f0 = f.value(x0)
    epsilon = Expression()

    x = x0 - gamma * g0

    # f^*(g0) = <g0;y>-f(y) for some y
    y = Point()
    fy = Expression()
    f.add_point((y, g0, fy))
    fstarg0 = g0 * y - fy

    # epsilon-subgradient condition:
    constraint = (f0 + fstarg0 - g0 * x0 <= epsilon)
    constraint.set_name("epsilon_subgradient({})_on_{}".format(f.get_name(), x0.get_name()))
    f.add_constraint(constraint)

    # Return the newly obtained point, the epsilon-subgradient, the value of f in x0, and epsilon.
    return x, g0, f0, epsilon
