from PEPit.point import Point
from PEPit.expression import Expression


def bregman_proximal_step(sx0, mirror_map, min_function, gamma):
    """
    This routine performs a proximal mirror step of step-size :math:`\\gamma`.
    That is, denoting :math:`f` the function to be minimized
    and :math:`h` the **mirror map**, it performs

    .. math:: x_{t+1} = \\arg\\min_x \\left[ f(x) + \\frac{1}{\\gamma} D_h(x; x_t) \\right]

    where :math:`D_h(x; x_t)` denotes the Bregman divergence of :math:`h` on :math:`x` with respect to :math:`x_t`.

    .. math:: D_h(x; x_t) \\triangleq h(x) - h(x_t) - \\left< \\nabla h(x_t) \\big| x - x_t \\right>

    Warning:
        The mirror map :math:`h` is assumed differentiable.

    Note:
        When :math:`h(x) = \\frac{1}{2}\\|x\\|^2`, :math:`D_h(x; x_t) = \\frac{1}{2}\\|x - x_t\\|^2`,
        and this step reduces to classical proximal step with step-size :math:`\\gamma`.

    By differentiating the previous objective function, one can observe that

    .. math:: \\nabla h(x_{t+1}) = \\nabla h(x_t) - \\gamma \\nabla f(x_{t+1})

    Args:
        sx0 (Point): starting gradient :math:`\\textbf{sx0} \\triangleq \\nabla h(x_t)`.
        mirror_map (Function): the reference function :math:`h` we computed Bregman divergence of.
        min_function (Function): function we aim to minimize.
        gamma (float): step size.

    Returns:
        x (Point): new iterate :math:`\\textbf{x} \\triangleq x_{t+1}`.
        sx (Point): :math:`h`'s gradient on new iterate :math:`x` :math:`\\textbf{sx} \\triangleq \\nabla h(x_{t+1})`.
        hx (Expression): :math:`h`'s value on new iterate :math:`\\textbf{hx} \\triangleq h(x_{t+1})`.
        gx (Point): :math:`f`'s gradient on new iterate :math:`x` :math:`\\textbf{gx} \\triangleq \\nabla f(x_{t+1})`.
        fx (Expression): :math:`f`'s value on new iterate :math:`\\textbf{fx} \\triangleq f(x_{t+1})`.

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
