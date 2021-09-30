from PEPit.point import Point
from PEPit.expression import Expression


def linearoptimization_step(dir, ind):
    """
    This routine performs a linear optimization step with objective function
     given by dir*x on the indicator function ind. That is, it evaluates
     x = argmin_{ind(x)} [dir*x]

    :param dir: gradient of the linear objective function
    :param ind: indicator function ind
    :return:  x = argmin_{ind(x)=0} [dir*x]
    """

    gx = -dir
    x = Point()
    fx = Expression()  # function value
    ind.add_point((x, gx, fx))

    return x, gx, fx
