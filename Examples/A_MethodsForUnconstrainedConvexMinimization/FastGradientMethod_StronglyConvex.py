from math import sqrt

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_fgm(mu, L, n, verbose=True):
    """
    Consider the convex minimization problem
        f_* = min_x f(x),
    where f is L-smooth and mu-strongly-convex.

    This code computes a worst-case guarantee for the fast gradient method, a.k.a. accelerated gradient method.
    That is, it computes the smallest possible tau(n, mu, L) such that the guarantee
        f(x_n) - f_* <= tau(n, mu, L) (f(x_0) -  f(x_*) +  mu/2*|| x_0 - x_* ||**2),
    is valid, where x_n is the output of the accelerated gradient method, and where x_* is a minimizer of f.

    In short, for given values of n and L, tau(n,mu,L) is be computed as the worst-case value of f(x_n)-f_* when
    (f(x_0) -  f(x_*) +  mu/2*|| x_0 - x_* ||**2) == 1.

    Theoretical rates can be found in the following paper [1,  Corollary 4.15]
    [1] Acceleration Methods, Monograph, Alexandre d’Aspremont, Damien Scieur, Adrien Taylor,
    https://arxiv.org/pdf/2101.09545.pdf

    :param mu: (float) the strong-convexity parameter.
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

    # Set the initial constraint that is a well-chosen distance between x0 and x^*
    problem.set_initial_condition(func.value(x0) - fs + mu / 2 * (x0 - xs) ** 2 <= 1)

    # Run n steps of the fast gradient method
    kappa = mu / L
    x_new = x0
    y = x0
    for i in range(n):
        x_old = x_new
        x_new = y - 1 / L * func.gradient(y)
        y = x_new + (1 - sqrt(kappa)) / (1 + sqrt(kappa)) * (x_new - x_old)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func.value(x_new) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    if mu > 0:
        theoretical_tau = (1 - sqrt(kappa)) ** n
    else:
        theoretical_tau = 0
        print("Momentum is tuned for strongly convex functions")

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the fast gradient method ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_*  <= {:.6} (f(x_0) -  f(x_*) +  mu/2*|| x_0 - x_* ||**2)'.format(
            pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_*  <= {:.6} (f(x_0) -  f(x_*) +  mu/2*|| x_0 - x_* ||**2)'.format(
            theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 2
    L = 1
    mu = 0.1

    pepit_tau, theoretical_tau = wc_fgm(mu=mu,
                                        L=L,
                                        n=n)
