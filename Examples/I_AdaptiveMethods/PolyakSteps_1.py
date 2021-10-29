from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_ps_1(L, mu, gamma, verbose=True):
    """
    Consider the minimization problem
        f_* = min_x f(x),
    where f is assumed L-smooth mu-strongly convex and x_* = argmin f(x) denotes the minimizer of f.

    This code computes a worst-case guarantee for a variant of Polyak steps.
    That is, it computes the smallest possible tau(n, L, mu) such that the guarantee
        ||x_{k+1} - x_*||^2 <= tau(n, L, mu) * ||x_{k+1} - x_*||^2
    is valid, where x_k is the output of Plyak steps.

    The detailed potential approach is available in [1, Proposition 1]
    [1] Mathieu Barre, Adrien Taylor, Alexandre d'Aspremont (2020).
      "Complexity Guarantees for Polyak Steps with Momentum."


    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param gamma: (float) the step size.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, param={'L': L, 'mu': mu})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value gn and fn
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Set the initial condition to the distance between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the Polayk steps at iteration 1
    x1 = x0 - gamma * g0
    _, _ = func.oracle(x1)

    # Set the initial condition to the Polyak step size
    problem.set_initial_condition(gamma * g0 ** 2 == 2 * (f0 - fs))

    # Set the performance metric to the distance between x_1 and x_* = xs
    problem.set_performance_metric((x1 - xs) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (gamma * L - 1) * (1 - gamma * mu) / (gamma * (L + mu) - 1)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of Polyak steps ***')
        print('\tPEP-it guarantee:\t\t||x_1 - x_*||^2  <= {:.6} ||x_0 - x_*||^2 '.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||x_1 - x_*||^2  <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    mu = 0.1
    gamma = 2 / (L + mu)

    pepit_tau, theoretical_tau = wc_ps_1(L=L,
                                         mu=mu,
                                         gamma=gamma)
