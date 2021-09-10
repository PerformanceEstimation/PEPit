from PEPit.point import Point


def inexactgradient(x0, f, epsilon, notion='absolute'):
    """
     TODO: CANEVAS DESCRIPTION DE CE QU'on FAIT


        :param x0:
        :param f:
        :param epsilon:
        :param notion:
        :return:
    """
    gx, fx = f.oracle(x0)
    gxeps = Point()
    if notion == 'absolute':
        f.add_constraint( (gx-gxeps)**2 - epsilon**2)
    elif notion == 'relative':
        f.add_constraint( (gx-gxeps)**2 - epsilon**2*(gx**2))

    return gxeps, fx
