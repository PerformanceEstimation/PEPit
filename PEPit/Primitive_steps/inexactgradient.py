from PEPit.point import Point


def inexactgradient(x0, f, epsilon, notion='absolute'):
    """
    This routines allows to return a point and its inexact gradient.

    Two notions of inaccuracy are proposed here. For an approximate subgradient g(x0) of f'(x0),
        - 'absolute' :
            ||g(x0) - f'(x0)||^2 <= epsilon^2,
        - 'relative' :
            ||g(x0) - f'(x0)||^2 <= epsilon^2 ||f'(x_0)||^2.

    :param x0 (Point): the starting point x0.
    :param f (function): the function on which the gradient is evaluated.
    :param epsilon (float): the required accuracy.
    :param notion (string): defines the mode 'absolute' or 'relative' inaccuracy (see above).

    :return:
        - gxeps (Point): the approximate (sub)gradient of at x0.
        - fx (Expression): the function f evaluated at x0.
    """
    gx, fx = f.oracle(x0)
    gxeps = Point()
    if notion == 'absolute':
        f.add_constraint((gx-gxeps)**2 - epsilon**2 <= 0)
    elif notion == 'relative':
        f.add_constraint((gx-gxeps)**2 - epsilon**2*(gx**2) <= 0)

    return gxeps, fx
