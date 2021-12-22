from PEPit.point import Point
from PEPit.expression import Expression


def bregman_gradient_step(gx0, sx0, mirror_map, gamma):
    """
    This routine performs a mirror step of step-size :math:`\\gamma`.
    That is, denoting :math:`f` the function to be minimized
    and :math:`h` the **mirror map**, it performs

    .. math:: x_{t+1} = \\arg\\min_x \\left[ f(x_t) + \\left< \\nabla f(x_t) \\big| x - x_t \\right>
              + \\frac{1}{\\gamma} D_h(x; x_t) \\right]

    where :math:`D_h(x; x_t)` denotes the Bregman divergence of :math:`h` on :math:`x` with respect to :math:`x_t`.

    .. math:: D_h(x; x_t) \\triangleq h(x) - h(x_t) - \\left< \\nabla h(x_t) \\big| x - x_t \\right>

    Warning:
        The mirror map :math:`h` is assumed differentiable.

    Note:
        When :math:`h(x) = \\frac{1}{2}\\|x\\|^2`, :math:`D_h(x; x_t) = \\frac{1}{2}\\|x - x_t\\|^2`,
        and this step reduces to classical gradient descent step with step-size :math:`\\gamma`.

    By differentiating the previous objective function, one can observe that

    .. math:: \\nabla h(x_{t+1}) = \\nabla h(x_t) - \\gamma \\nabla f(x_t)

    Args:
        sx0 (Point): starting gradient :math:`\\textbf{sx0} \\triangleq \\nabla h(x_t)`.
        gx0 (Point): descent direction :math:`\\textbf{gx0} \\triangleq \\nabla f(x_t)`.
        mirror_map (Function): the reference function :math:`h` we computed Bregman divergence of.
        gamma (float): step size.

    Returns:
        x (Point): new iterate :math:`\\textbf{x} \\triangleq x_{t+1}`.
        sx (Point): :math:`h`'s gradient on new iterate :math:`x` :math:`\\textbf{sx} \\triangleq \\nabla h(x_{t+1})`.
        hx (Expression): :math:`h`'s value on new iterate :math:`\\textbf{hx} \\triangleq h(x_{t+1})`.

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
