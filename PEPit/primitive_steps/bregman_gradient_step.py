from PEPit.point import Point
from PEPit.expression import Expression


def bregman_gradient_step(gx0, sx0, mirror_map, gamma):
    """
    This routine outputs :math:`x` by performing a mirror step of step-size :math:`\\gamma`.
    That is, denoting :math:`f` the function to be minimized
    and :math:`h` the **mirror map**, it performs

    .. math:: x = \\arg\\min_x \\left[ f(x_0) + \\left< \\nabla f(x_0);\, x - x_0 \\right>
              + \\frac{1}{\\gamma} D_h(x; x_0) \\right],

    where :math:`D_h(x; x_0)` denotes the Bregman divergence of :math:`h` on :math:`x` with respect to :math:`x_0`.

    .. math:: D_h(x; x_0) \\triangleq h(x) - h(x_0) - \\left< \\nabla h(x_0);\, x - x_0 \\right>.

    Warning:
        The mirror map :math:`h` is assumed differentiable.

    By differentiating the previous objective function, one can observe that

    .. math:: \\nabla h(x) = \\nabla h(x_0) - \\gamma \\nabla f(x_0).

    Args:
        sx0 (Point): starting gradient :math:`\\textbf{sx0} \\triangleq \\nabla h(x_0)`.
        gx0 (Point): descent direction :math:`\\textbf{gx0} \\triangleq \\nabla f(x_0)`.
        mirror_map (Function): the reference function :math:`h` we computed Bregman divergence of.
        gamma (float): step size.

    Returns:
        x (Point): new iterate :math:`\\textbf{x} \\triangleq x`.
        sx (Point): :math:`h`'s gradient on new iterate :math:`x` :math:`\\textbf{sx} \\triangleq \\nabla h(x)`.
        hx (Expression): :math:`h`'s value on new iterate :math:`\\textbf{hx} \\triangleq h(x)`.

    """

    # Instantiating point and function value.
    x = Point()
    hx = Expression()

    # Apply Bregman gradient step.
    sx = sx0 - gamma * gx0

    # Store triplet in mirror map list of points.
    mirror_map.add_point((x, sx, hx))

    # Return the aforementioned triplet.
    return x, sx, hx
