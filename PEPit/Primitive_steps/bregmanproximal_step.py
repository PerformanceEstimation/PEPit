from PEPit.point import Point
from PEPit.expression import Expression

def BregmanProximal_Step(sx0, mirror_map, min_function, gamma):
    """
    [x,gx,hx] = mirror_prox(sx0, mirror_map, min_function, gamma, tag)

    This routine performs a proximal mirror step of step size gamma.

    That is, denoting by h(.) the mirror map, and f(.) the function to be
    minimized, it performs
    h'(x) = h'(x0) - gamma*f'(x), where h'(x) and h'(x0) are respectively
                                 (sub)gradients of the mirror map at x and x0,
                                 and f'(x) is a subgradient of f at x.

    :param sx0: starting gradient sx0 (e.g., gradient at x0 of 'mirror_map'),
    :param mirror_map:mirror_map on which the (sub)gradient will be evaluated,
    :param min_function: min_function which we aim to minimize,
    :param gamma: step size.
    :return:
            - x:  mirror point,
            - sx: subgradient of the mirror_map at x that was used in the procedure,
            - hx: value of the mirror_map evaluated at x,
            - gx: subgradient of the min_function at x that was used in the procedure,
            - fx: value of the min_function evaluated at x.
    """

    hx = Expression()
    x = Point()
    fx = Expression()
    gx = Point()

    sx = sx0 - gamma * gx
    mirror_map.add_point((x, sx, hx))
    min_function.add_point((x, gx, fx))
    return x, sx, hx, gx, fx