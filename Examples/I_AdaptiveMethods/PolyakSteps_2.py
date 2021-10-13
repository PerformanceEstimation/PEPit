import cvxpy as cp

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction



def wc_ps_2(L, mu, gamma, verbose=True):
    """
    Consider the minimization problem
        f_* = min_x f(x),
    where f is assumed L-smooth mu-strongly convex and x_* = argmin f(x) denotes the minimizer of f.

    This code computes a worst-case guarantee for a variant of Polyak steps.
    That is, it computes the smallest possible tau(n, L, mu) such that the guarantee
        f(x_{k+1}) - f_* <= tau(n, L, mu) * (f(x_k) - f_*)
    is valid, where x_k is the output of Plyak steps.

    The detailed potential approach is available in [1, Proposition 2]
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
    func = problem.declare_function(SmoothStronglyConvexFunction,
                                    {'L': L, 'mu': mu})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value gn and fn
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Set the initial condition to the distance betwenn x0 and xs
    problem.set_initial_condition(f0 - fs <= 1)

    # Run the Polayk steps at iteration 1
    x1 = x0 - gamma * g0
    g1, f1 = func.oracle(x1)

    # Set the initial condition to the Polyak step size
    problem.set_initial_condition(g0 ** 2 == 2*L*(2 - gamma) * (f0 - fs))

    # Set the performance metric to the distance in function values between x_1 and x_* = xs
    problem.set_performance_metric(f1 - fs)

    # Solve the PEP
    pepit_tau = problem.solve(solver=cp.MOSEK, verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (gamma * L - 1) * (L * gamma * (3 - gamma * (L + mu)) - 1)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of Polay steps ***')
        print('\tPEP-it guarantee:\t\t f(x_1) - f_*  <= {:.6} (f(x_0) - f_*) '.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_1) - f_*  <= {:.6} (f(x_0) - f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    mu = 0.1
    gamma = 2/(L + mu)

    pepit_tau, theoretical_tau = wc_ps_2(L=L,
                                        mu=mu,
                                        gamma=gamma)