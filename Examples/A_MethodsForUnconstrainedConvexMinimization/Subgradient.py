import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.convex_lipschitz_function import ConvexLipschitzFunction


def wc_subgd(M, N, gamma, verbose=True):
    """
    Consider the minimization problem
        f_* = min_x f(x),
    where f is convex and M-Lipschitz. This problem is a non-smooth minimization problem.

    This code computes a worst-case guarantee for the subgradient method. That is, it computes
    the smallest possible tau(n, M) such that the guarantee
        min_{0 \leq i \leq N} f(x_i) - f_* <= tau(n, M) * ||x_0 - x_*||^2
    is valid, where x_n is the output of the gradient descent with exact linesearch,
    and where x_* is the minimizer of f.

    We show how to compute the worst-case value of min_i F(xi)-F(xs) when xi is
    obtained by doing i steps of a subgradient method starting with an initial
    iterate satisfying ||x0-xs||<=1.

    :param M: (float) the lipschitz parameter.
    :param N: (int) the number of iterations
    :param gamma: optimal step size is 1/(sqrt(N+1)*M)
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex lipschitz function
    func = problem.declare_function(ConvexLipschitzFunction,
                                    {'M': M})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the subgradient method
    x = x0
    gx, fx = func.oracle(x)

    for _ in range(N):
        problem.set_performance_metric(fx - fs)
        x = x - gamma * gx
        gx, fx = func.oracle(x)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_tau = problem.solve()

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = M / np.sqrt(N + 1)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of sugbradient method ***')
        print('\tPEP-it guarantee:\t\t min_(0 \leq i \leq N) f(x_i) - f_*  <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t min_(0 \leq i \leq N) f(x_i) - f_*  <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    M = 2
    N = 6
    gamma = 1 / (np.sqrt(N + 1) * M)

    rate = wc_subgd(M=M,
                    N=N,
                    gamma=gamma)
