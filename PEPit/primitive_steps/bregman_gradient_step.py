from PEPit.point import Point
from PEPit.expression import Expression


def bregman_gradient_step(gx0, sx0, mirror_map, gamma):
    """
    [x,gx,hx] = mirror(gx0, sx0, func, gamma, tag)

    This routine performs a mirror step of step size gamma.
    That is, denoting by h(.) the mirror map, it performs
    h'(x) = h'(x0) - gamma*gx0, where h'(x) and h'(x0) are respectively gradients of the mirror map at x and x0.

    NOTE: it assumes the mirror map is differentiable.

    Args:
        sx0: starting gradient sx0 (e.g., gradient at x0 of 'mirror_map'),
        gx0: step gx0 (e.g., gradient at x0 of the function to be minimized),
        mirror_map: on which the (sub)gradient will be evaluated,
        gamma: step size.

    Returns:
        x:  mirror point,
        sx: subgradient of the mirror_map at x that was used in the procedure,
        hx: value of the mirror_map evaluated at x.
    """

    hx = Expression()
    x = Point()

    sx = sx0 - gamma * gx0
    mirror_map.add_point((x, sx, hx))

    return x, sx, hx
