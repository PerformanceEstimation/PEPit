from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_gd(mu, L, gamma, n, verbose=True):
    """
    Consider the minimization problem
        f_* = min_x f(x),
    where f is L-smooth and mu-strongly convex.
    This code computes a worst-case guarantee for the gradient method with fixed step size. That is, it computes
    the smallest possible tau(n, L, mu) such that the guarantee
        f(x_n) - f_* <= tau(n, L, mu) * || x_0 - x_* ||^2
    is valid, where x_n is the output of the gradient descent with fixed step size,
    and where x_* is the minimizer of f.

    Result to be compared with that of
    [1] Yoel Drori. "Contributions to the Complexity Analysis of
        Optimization Algorithms." PhD thesis, Tel-Aviv University, 2014.

    :param mu: (float) the strong convexity parameter.
    :param L: (float) the smoothness parameter.
    :param gamma: (float) step size.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction,
                                    {'mu': mu, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the GD method
    x = x0
    for _ in range(n):
        x = x - gamma * func.gradient(x)

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric(func.value(x) - fs)

    # Solve the PEP
    pepit_tau = problem.solve()

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L/2/(2*n+1)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of gradient descent with fixed step sizes ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 2
    mu = 0
    L = 1
    gamma = 1/L

    wc = wc_gd(mu=mu,
               L=L,
               gamma=gamma,
               n=n)
