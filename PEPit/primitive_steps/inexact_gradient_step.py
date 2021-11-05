from PEPit.primitive_steps.inexact_gradient import inexact_gradient


def inexact_gradient_step(x0, f, gamma, epsilon, notion='absolute'):
    """
    This routines allows to evaluated an inexact (sub)gradient.

    :param x0 (Point): starting point x0.
    :param f (Function): a function.
    :param gamma (float): the step size parameter.
    :param epsilon (float): the required accuracy.
    :param notion (string): defines the mode (absolute or relative inaccuracy, see in inexact_gradient_descent.py).

    :return:
        - x (Point) the output point.
        - gx (Point) the approximate (sub)gradient of f at x.
        - fx (Expression) the function f evaluated at x.
    """
    dx0, fx0 = inexact_gradient(x0, f, epsilon, notion=notion)
    x = x0 - gamma * dx0

    return x, dx0, fx0
