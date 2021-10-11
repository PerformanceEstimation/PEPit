from math import sqrt

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_convex_function import SmoothConvexFunction
from PEPit.Primitive_steps.exactlinesearch_step import exactlinesearch_step


def wc_CG(L, n):
    """
    Consider the convex minimization problem
        f_* = min_x f(x),
    where f is L-smooth and convex.
    This code computes a worst-case guarantee for the conjugate gradient (CG) method (with exact span searches).
    That is, it computes the smallest possible tau(n,L) such that the guarantee
        f(x_n) - f_* <= tau(n,L) * ||x_0-x_*||^2
    is valid, where x_n is the output of the conjugate gradient method, and where x_* is a minimizer of f.

    In short, for given values of n and L, tau(n,L) is be computed as the worst-case value of f(x_n)-f_* when
    ||x_0 - x_* || == 1.

    The detailed approach (based on convex relaxations) is available in
    [1] Y. Drori and A. Taylor (2020). Efficient first-order methods for convex minimization: a constructive approach.
    Mathematical Programming 184 (1), 183-220.

    The tight guarantee obtained in [1] is tau(n,L)

    :param L: (float) the smoothness parameter.
    :param n: (int) number of iterations.

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothConvexFunction, {'L': L})

    # Start by defining its unique optimal point
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the GD method with ELS
    x_new = x0
    g0, f0 = func.oracle(x0)
    span = [g0]  # list of search directions
    for i in range(n):
        x_old = x_new
        x_new, gx, fx = exactlinesearch_step(x_new, func, span)
        span.append(gx)
        span.append(x_old - x_new)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    wc = problem.solve()
    # Theoretical guarantee (for comparison)
    theta_new = 1
    for i in range(n):
        if i < n - 1:
            theta_new = (1 + sqrt(4 * theta_new ** 2 + 1)) / 2
        else:
            theta_new = (1 + sqrt(8 * theta_new ** 2 + 1)) / 2
    theory = L / 2 / theta_new ** 2

    print('*** Example file: worst-case performance of conjugate gradient (CG) with exact span searches ***')
    print('\tPEP-it guarantee:\t f(y_n)-f_* <= ', wc)
    print('\tTheoretical guarantee:\t f(y_n)-f_* <= ', theory)
    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return wc, theory


if __name__ == "__main__":
    n = 2
    L = 1
    wc, theory = wc_CG(L=L, n=n)
