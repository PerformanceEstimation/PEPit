import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_convex_function import SmoothConvexFunction
from PEPit.Function_classes.convex_indicator import ConvexIndicatorFunction
from PEPit.Primitive_steps.linearoptimization_step import linearoptimization_step


def wc_cg_fw(L, D, n):
    """
    In this example, we use a conditional gradient method for
    solving the constrained smooth convex minimization problem
        min_x { F(x) = f_1(x) + f_2(x) }
    for notational convenience we denote xs=argmin_x F(x);
    where f_1(x) is L-smooth and convex and where f_2(x) is
    a convex indicator function of diameter at most D.

    We show how to compute the worst-case value of F(xN)-F(xs) when xN is
    obtained by doing N steps of the method starting with a feasible point

    The theoretical guarantee is presented in the following reference.
    [1] Jaggi, Martin. "Revisiting Frank-Wolfe: Projection-free sparse
     convex optimization." In: Proceedings of the 30th International
     Conference on Machine Learning (ICML-13), pp. 427–435 (2013)

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param alpha: (float) parameter of the scheme.
    :param theta: (float) parameter of the scheme.
    :param n: (int) number of iterations.
    :return:
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex lipschitz function
    func1 = problem.declare_function(SmoothConvexFunction,
                                     {'L': L})
    func2 = problem.declare_function(ConvexIndicatorFunction,
                                     {'D': D})
    func = func1 + func2

    # Start by defining its unique optimal point and its function value
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute trajectory starting from x0
    x = x0
    for i in range(n):
        g = func1.gradient(x)
        y, _, _ = linearoptimization_step(g, func2)
        lam = 2 / (i + 1)
        x = (1 - lam) * x + lam * y

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric((func.value(x)) - fs)

    # Solve the PEP
    wc = problem.solve()

    # Theoretical guarantee (for comparison)
    # when theta = 1
    theory = 2 * L * D ** 2 / (n + 2)
    print('*** Example file: worst-case performance of the Conditional Gradient (Franck-Wolfe) in function value ***')
    print('\tPEP-it guarantee:\tf(y_n) - f_* <= ', wc)
    print('\tTheoretical standard guarantee :\tf(y_n) - f_* <=  <= ', theory)
    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)

    # Return the rate of the evaluated method
    return wc


if __name__ == "__main__":
    D = 1.
    L = 1.
    n = 10

    rate = wc_cg_fw(L=L,
                    D=D,
                    n=n)
