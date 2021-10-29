import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_sgd(L, mu, gamma, v, R, n, verbose=True):
    """
    Consider the finite sum minimization problem
        f_* = min_x {F(x) = 1/n [f1(x) + ... + fn(x)]},
    where f1, ..., fn are assumed L-smooth and mu-strongly convex.

    In addition, we assume a bounded variance at the optimal point :
        \mathbb{E}(sum_i(||fi'(x^*)||^2)/n) <= v^2,
    which is standard from the SGD literature.

    This code computes a worst-case guarantee for one step of the stochastic gradient descent in expectation,
    for the distance to optimality.

    That is, it computes the smallest possible tau(n,L,mu,epsilon) such that the guarantee
    \mathbb{E}[||x_1 - x^*||^2] <= tau(L, mu, gamma, v, R, n) * (f(x_0) - f_*)
    is valid, where x_1 is the output of one step of stochastic gradient descent: x_1 = x_0 - \gamma f'_{i_0}(x_0),
    with i_0 uniformly sampled in {1, \dots, n}, and the expectation is taken over the randomness of i_0.

    We will observe it does not depend on the number n of functions for this particular setting,
    meaning that the guarantees are also valid for expectation minimization settings (i.e., when n goes to infinity).

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param gamma: (float) the step size.
    :param v: (float) the variance bound.
    :param R: (float) the initial distance.
    :param n: (int) number of functions.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    fn = [problem.declare_function(SmoothStronglyConvexFunction, param={'L': L, 'mu': mu}) for _ in range(n)]
    func = np.mean(fn)

    # Start by defining its unique optimal point xs = x_*
    xs = func.optimal_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the bounded variance and the distance between initial point and optimal one
    var = np.mean([f.gradient(xs)**2 for f in fn])

    problem.set_initial_condition(var <= v**2)
    problem.set_initial_condition((x0 - xs)**2 <= R**2)

    # Run n-step of the stochastic gradient and compute the averaged distance to optimality
    distavg = np.mean([(x0 - gamma * f.gradient(x0) - xs)**2 for f in fn])

    # Set the performance metric to the distance average to optimal point
    problem.set_performance_metric(distavg)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    kappa = L/mu
    theoretical_tau = 1/2*(1-1/kappa)**2*R**2 + 1/2*(1-1/kappa)*R*np.sqrt((1-1/kappa)**2*R**2 + 4*v**2/L**2) + v**2/L**2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of stochastic gradient descent with fixed step size ***')
        print('\tPEP-it guarantee:\t\t sum_i(||x_i - x_*||^2)/n <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t sum_i(||x_i - x_*||^2)/n <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 5
    L = 1
    mu = 0.1
    v = 1
    R = 2
    gamma = 1/L

    pepit_tau, theoretical_tau = wc_sgd(L=L,
                                        mu=mu,
                                        gamma=gamma,
                                        v=v,
                                        R=R,
                                        n=n)
