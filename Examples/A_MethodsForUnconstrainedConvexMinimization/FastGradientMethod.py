from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_fgm(mu, L, n, verbose=True):
    """
    Consider the convex minimization problem
        f_* = min_x f(x),
    where f is L-smooth (and possibly mu-strongly-convex).

    This code computes a worst-case guarantee for the fast gradient method, a.k.a. accelerated gradient method.
    That is, it computes the smallest possible tau(n,L,mu) such that the guarantee
        f(x_n) - f_* <= tau(n,L,mu) * || x_0 - x_* ||^2,
    is valid, where x_n is the output of the accelerated gradient method, and where x_* is a minimizer of f.

    In short, for given values of n and L, tau(n,L,mu) is be computed as the worst-case value of f(x_n)-f_* when
    || x_0 - x_* || == 1.

    Theoretical rates can be found in the following paper
    For an Upper bound (not tight)
    [1] A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems∗
    Amir Beck and Marc Teboulle

    For an exact bound (convex):
    [2] Exact Worst-case Performance of First-order Methods for Composite Convex Optimization
    Adrien B. Taylor, Julien M. Hendrickx, François Glineur

    :param L: (float) the smoothness parameter.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the fast gradient method
    x_new = x0
    y = x0
    for i in range(n):
        x_old = x_new
        x_new = y - 1 / L * func.gradient(y)
        y = x_new + i / (i + 3) * (x_new - x_old)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func.value(x_new) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Theoretical guarantee (for comparison)
    if mu == 0:
        theoretical_tau = 2 * L / (n ** 2 + 5 * n + 6)  # tight, see [2], Table 1 (column 1, line 1)
    else:
        theoretical_tau = 2 * L / (n ** 2 + 5 * n + 6)  # not tight (bound for smooth convex functions)
        print('Warning: momentum is tuned for non-strongly convex functions.')

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of conjugate gradient method ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 1
    L = 1
    mu = 0

    pepit_tau, theoretical_tau = wc_fgm(mu=mu, L=L, n=n)
