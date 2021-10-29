from PEPit.point import Point


def exactlinesearch_step(x0, f, directions):
    """
    This routines *mimics* an exact line search.
    Indeed, at each iteration k:
        gamma = argmin_t f(x_k - tf'(x_k))
        x_{k+1} = x_k - gamma * f(x_k)
    The two iterated are thus defined by the following two conditions :
        x_{k+1} - x_{k} + gamma f'(x_k) = 0
        f'(x_{k+1})^T(x_{k+1} - x_k) = 0
    In this routine, we define each new triplet (x_{k+1}, f'(x_{k+1}), f(x_{k+1})) such that
        f'(x_{k+1})^Tf'(x_k) = 0

    :param x0 (Point): the starting point.
    :param f (function): the function on which the (sub)gradient will be evaluated.
    :param directions: (List of Points) the list of all directions required to be orthogonal to the
                        (sub)gradient of x. Note that (x-x0) is automatically constrained to be orthogonal
                        to the subgradient of f at x.

    :return:
            - x (Point), such that all vectors in directions are orthogonal to the (sub)gradient of f at x.
            - gx (Point), a (sub)gradient of f at x.
            - fx (Expression), the function f evaluated at x.
    """

    x = Point()
    gx, fx = f.oracle(x)
    f.add_constraint((x - x0) * gx == 0)
    for d in directions:
        f.add_constraint(d * gx == 0)

    return x, gx, fx
