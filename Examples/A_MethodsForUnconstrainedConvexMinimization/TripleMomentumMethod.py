import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_tmm(mu, L, n, verbose=True):
    """
    Consider the convex minimization problem
        f_* = min_x f(x),
    where f is L-smooth and mu-strongly-convex.

    This code computes a worst-case guarantee for the Triple Momentum Method.
    That is, it computes the smallest possible tau(n, mu, L) such that the guarantee
        f(x_n) - f_* <= tau(n, mu, L) ||x_0 - x_*||^2,
    is valid, where x_n is the output of the triple momentum method, and where x_* is a minimizer of f.

    [1] Van Scoy, B., Freeman, R. A., & Lynch, K. M. (2018).
    "The fastest known globally convergent first-order method for
    minimizing strongly convex functions."
    IEEE Control Systems Letters, 2(1), 49-54.

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex
    func = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Set the parameters of the Triple Momentum Method
    kappa = L / mu
    rho = (1 - 1 / np.sqrt(kappa))
    alpha = (1 + rho) / L
    beta = rho ** 2 / (2 - rho)
    gamma = rho ** 2 / (1 + rho) / (2 - rho)
    delta = rho ** 2 / (1 - rho ** 2)

    # Run n steps of the Triple Momentum Method
    x_old = x0
    x_new = x0
    y = x0
    for _ in range(n + 1):
        x_inter = (1 + beta) * x_new - beta * x_old - alpha * func.gradient(y)
        y = (1 + gamma) * x_inter - gamma * x_new
        x = (1 + delta) * x_inter - delta * x_new
        x_new, x_old = x_inter, x_new

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func.value(x) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = rho ** (2 * (n + 1)) * L / 2 * kappa

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Triple Momentum Method ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    mu = 0.1
    L = 1.
    n = 4

    pepit_tau, theoretical_tau = wc_tmm(mu=mu,
                                        L=L,
                                        n=n)
