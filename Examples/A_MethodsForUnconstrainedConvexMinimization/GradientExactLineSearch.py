from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Primitive_steps.exactlinesearch_step import exactlinesearch_step


def wc_els(L, mu, n, verbose=True):
    """
    Consider the minimization problem
        f_* = min_x f(x),
    where f is L-smooth and mu-strongly convex.
    This code computes a worst-case guarantee for the gradient method with exact linesearch (ELS). That is, it computes
    the smallest possible tau(n, L, mu) such that the guarantee
        f(x_n) - f_* <= tau(n, L, mu) * (f(x_0) - f_*)
    is valid, where x_n is the output of the gradient descent with exact linesearch,
    and where x_* is the minimizer of f.

    In short, for given values of n and L and mu, tau(n, L, mu) is computed as the worst-case value of f(x_n)-f_* when
    f(x_0) - f_* <= 1.

    The detailed approach (based on convex relaxations) is available in
    [1] De Klerk, Etienne, François Glineur, and Adrien B. Taylor.
    "On the worst-case complexity of the gradient method with exact line search for smooth strongly convex functions."
    Optimization Letters (2017).

    The tight guarantee obtained in [1, Theorem 1.2] is tau(n, L, mu) = ((L-mu)/(L+mu))^(2n).

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, {'mu': mu, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value g0 and f0
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Set the initial constraint that is the difference between f0 and f_*
    problem.set_initial_condition(f0 - fs <= 1)

    # Run n steps of GD method with ELS
    x = x0
    gx = g0
    for i in range(n):
        x, gx, fx = exactlinesearch_step(x, func, [gx])

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = ((L - mu) / (L + mu)) ** (2 * n)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of gradient descent with exact linesearch (ELS) ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 2
    L = 1
    mu = .1

    pepit_tau, theoretical_tau = wc_els(L=L, mu=mu, n=n)
