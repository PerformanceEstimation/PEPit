from PEPit.point import Point
from PEPit.expression import Expression


def linear_optimization_step(dir, ind):
    """
    This routine performs a linear optimization step with objective function
     given by dir*x on the indicator function ind. That is, it evaluates
     x = argmin_{ind(x)} [dir*x]

    :param dir (Point): gradient of the linear objective function
    :param ind (function): indicator function ind
    :return:
        - x (Point) : the point x.
        - gx (Point) : the (sub)gradient of f at x.
        - fx (Expression) : the function f evaluated at x.
    """

    gx = -dir
    x = Point()
    fx = Expression()  # function value
    ind.add_point((x, gx, fx))

    return x, gx, fx
