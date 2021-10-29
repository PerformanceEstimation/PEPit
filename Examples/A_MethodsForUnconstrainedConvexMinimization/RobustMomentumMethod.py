import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_rmm(mu, L, lam, verbose=True):
    """
    Consider the convex minimization problem
        f_* = min_x f(x),
    where f is L-smooth and mu-strongly-convex.

    This code computes a worst-case guarantee for the robust momentum.
    That is, it computes the smallest possible tau(n, mu, L) such that the guarantee
        v_(x_{n+1}) <= tau(n, mu, L) v_(x_{n}),
    is valid, where x_n is the output of the optimized gradient method, where x_* is a minimizer of f,
    and where v(x_n) is a well-chosen Lyapunov function decreasing along the sequence :

    We show how to compute the tight rate for the Lyapunov function developped in
    [1] Cyrus, S., Hu, B., Van Scoy, B., & Lessard, L. "A robust accelerated
         optimization algorithm for strongly convex functions." In 2018 Annual
         American Control Conference (ACC) (pp. 1376-1381). IEEE.

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param lam: (float) if lam=1 it is the gradient descent, if lam=0, it is the Triple Momentum Method.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then Define the starting points of the algorithm, x0 and x1
    x0 = problem.set_initial_point()
    x1 = problem.set_initial_point()

    # Set the parameters of the robust momentum method
    kappa = L / mu
    rho = lam * (1 - 1 / kappa) + (1 - lam) * (1 - 1 / float(np.sqrt(kappa)))
    alpha = kappa * (1 - rho) ** 2 * (1 + rho) / L
    beta = kappa * rho ** 3 / (kappa - 1)
    gamma = rho ** 3 / ((kappa - 1) * (1 - rho) ** 2 * (1 + rho))
    l = mu ** 2 * (kappa - kappa * rho ** 2 - 1) / (2 * rho * (1 - rho))
    nnu = (1 + rho) * (1 - kappa + 2 * kappa * rho - kappa * rho ** 2) / (2 * rho)

    # Run one step of the Robust Momentum Method
    y0 = x1 + gamma * (x1 - x0)
    g0, f0 = func.oracle(y0)
    x2 = x1 + beta * (x1 - x0) - alpha * g0
    y1 = x2 + gamma * (x2 - x1)
    g1, f1 = func.oracle(y1)
    x3 = x2 + beta * (x2 - x1) - alpha * g1

    z1 = (x2 - (rho ** 2) * x1) / (1 - rho ** 2)
    z2 = (x3 - (rho ** 2) * x2) / (1 - rho ** 2)

    # Evaluate the lyapunov function at the first and second iterates
    q0 = (L - mu) * (f0 - fs - mu / 2 * (y0 - xs) ** 2) - 1 / 2 * (g0 - mu * (y0 - xs)) ** 2
    q1 = (L - mu) * (f1 - fs - mu / 2 * (y1 - xs) ** 2) - 1 / 2 * (g1 - mu * (y1 - xs)) ** 2
    initLyapunov = l * (z1 - xs) ** 2 + q0
    finalLyapunov = l * (z2 - xs) ** 2 + q1

    # Set the initial constraint that is a bound on the initial Lyapunov function
    problem.set_initial_condition(initLyapunov <= 1)

    # Set the performance metric to the final distance to optimum, that is the final Lyapunov function
    problem.set_performance_metric(finalLyapunov)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = rho ** 2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Robust Momentum Method ***')
        print('\tPEP-it guarantee:\t\t v(x_(n+1)) <= {:.6} v(x_n)'.format(
            pepit_tau))
        print('\tTheoretical guarantee:\t v(x_(n+1)) <= {:.6} v(x_n)'.format(
            theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    mu = 0.1
    L = 1.
    lam = 0.2

    pepit_tau, theoretical_tau = wc_rmm(mu=mu,
                                        L=L,
                                        lam=lam)
