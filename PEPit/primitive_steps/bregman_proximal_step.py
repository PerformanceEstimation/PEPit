from PEPit.point import Point
from PEPit.expression import Expression


def bregman_proximal_step(sx0, mirror_map, min_function, gamma):
    """
    This routine outputs :math:`x` by performing a proximal mirror step of step-size :math:`\\gamma`.
    That is, denoting :math:`f` the function to be minimized
    and :math:`h` the **mirror map**, it performs

    .. math:: x = \\arg\\min_x \\left[ f(x) + \\frac{1}{\\gamma} D_h(x; x_0) \\right],

    where :math:`D_h(x; x_0)` denotes the Bregman divergence of :math:`h` on :math:`x` with respect to :math:`x_0`.

    .. math:: D_h(x; x_0) \\triangleq h(x) - h(x_0) - \\left< \\nabla h(x_0);\, x - x_0 \\right>.

    Warning:
        The mirror map :math:`h` is assumed differentiable.

    By differentiating the previous objective function, one can observe that

    .. math:: \\nabla h(x) = \\nabla h(x_0) - \\gamma \\nabla f(x).

    Args:
        sx0 (Point): starting gradient :math:`\\textbf{sx0} \\triangleq \\nabla h(x_0)`.
        mirror_map (Function): the reference function :math:`h` we computed Bregman divergence of.
        min_function (Function): function we aim to minimize.
        gamma (float): step size.

    Returns:
        x (Point): new iterate :math:`\\textbf{x} \\triangleq x`.
        sx (Point): :math:`h`'s gradient on new iterate :math:`x` :math:`\\textbf{sx} \\triangleq \\nabla h(x)`.
        hx (Expression): :math:`h`'s value on new iterate :math:`\\textbf{hx} \\triangleq h(x)`.
        gx (Point): :math:`f`'s gradient on new iterate :math:`x` :math:`\\textbf{gx} \\triangleq \\nabla f(x)`.
        fx (Expression): :math:`f`'s value on new iterate :math:`\\textbf{fx} \\triangleq f(x)`.

    """

    # Instantiate new point
    x = Point()

    # Create f's gradient and function value on x
    gx = Point()
    fx = Expression()

    # Create h's gradient and function value on x
    sx = sx0 - gamma * gx
    hx = Expression()

    # Add triplets to lists of points
    min_function.add_point((x, gx, fx))
    mirror_map.add_point((x, sx, hx))

    # Return all 5 new elements
    return x, sx, hx, gx, fx
