from PEPit.point import Point


def exactlinesearch_step(x0, f, directions):
    """
    TODO: CANEVAS DESCRIPTION DE CE QU'on FAIT
    Output the proximal step...


    :param mu: (float) the strong convexity parameter.
    :param L: (float) the smoothness parameter.
    :param gamma: (float) step size.
    :param n: (int) number of iterations.
    :return:
    """

    x = Point()
    gx, fx = f.oracle(x)
    f.add_constraint((x - x0) * gx <= 0)
    f.add_constraint(-(x - x0) * gx <= 0)
    for d in directions:
        f.add_constraint(d * gx <= 0)
        f.add_constraint(-d * gx <= 0)

    return x, gx, fx
