from math import sqrt

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_convex_function import SmoothConvexFunction
from PEPit.Primitive_steps.exactlinesearch_step import exactlinesearch_step


def wc_CG(L, n, verbose=True):
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
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction, {'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the Conjugate Gradient method
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
    pepit_tau = problem.solve()

    # Compute theoretical guarantee (for comparison)
    theta_new = 1
    for i in range(n):
        if i < n - 1:
            theta_new = (1 + sqrt(4 * theta_new ** 2 + 1)) / 2
        else:
            theta_new = (1 + sqrt(8 * theta_new ** 2 + 1)) / 2
    theoretical_tau = L / 2 / theta_new ** 2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of conjugate gradient method ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 2
    L = 1

    pepit_tau, theoretical_tau = wc_CG(L=L,
                                        n=n)
