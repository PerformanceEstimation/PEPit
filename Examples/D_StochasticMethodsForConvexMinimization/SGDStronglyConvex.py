import cvxpy as cp
import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction



def wc_sgd(L, mu, gamma, v, R, n, verbose=True):
    """
    Consider the finite sum minimization problem
        f_* = min_x F(x) = 1/n [f1(x) + ... + fn(x)],
    where f1, ..., fn are assumed L-smooth and mu-strongly convex.

    In addition, we assume a bounded variance at optimality :
        E||fi'x^*)||^2 <= v^2,
    which is standard from the SGD literature.

    This code computes a worst-case guarantee for the stochastic gradient descent, for the distance
    to optimality. We will observe it does not depend on n for this particular setting, meaning that
    the guarantees are also valid for expectation minimization settings (i.e., when n goes to infinity).

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param gamma: (float) the step size.
    :param v: (float) the variance bound.
    :param R: (float) the initial distance.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    f = problem.declare_function(SmoothStronglyConvexFunction,
                                    {'L': L, 'mu': mu})
    func = f/n
    fn = [f for i in range(n+1)]
    for i in range(n-1):
        fn[i+1] = problem.declare_function(SmoothStronglyConvexFunction,
                                           {'L': L, 'mu': mu})
        func += fn[i+1]/n

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value g0 and f0
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Set the initial constraint that is the bounded variance and the distance between initial point and optimal one
    var = fn[0].gradient(xs)**2
    for i in range(1, n+1):
        gx = fn[i].gradient(xs)
        var = var + gx**2
    var = var/n
    problem.set_initial_condition(var <= v**2)
    problem.set_initial_condition((x0 - xs)**2 <= R**2)

    # Run n-step of the stochastic gradient and compute the averaged distance to optimality
    distavg = (x0 - gamma * fn[i].gradient(x0) - xs)**2/n
    for i in range(1, n):
        x = x0 - gamma * fn[i].gradient(x0)
        distavg = distavg + (x - xs)**2/n

    # Set the performance metric to the distance average to optimal point
    problem.set_performance_metric(distavg)

    # Solve the PEP
    pepit_tau = problem.solve(solver=cp.MOSEK, verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    kappa = L/mu
    theoretical_tau = 1/2*(1-1/kappa)**2*R**2 + 1/2*(1-1/kappa)*R*np.sqrt((1-1/kappa)**2*R**2 + 4*v**2/L**2) + v**2/L**2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of stochastic gradient descent with fixed step size ***')
        print('\tPEP-it guarantee:\t\t sum_i((x_i - x_*)^2)/n <= {:.6} (x0 - x_*)^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t sum_i((x_i - x_*)^2)/n <= {:.6} (x0 - x_*)^2'.format(theoretical_tau))

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