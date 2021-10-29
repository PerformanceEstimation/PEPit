from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Primitive_steps.inexactgradient import inexactgradient


def wc_InexactGrad(L, mu, epsilon, n, verbose=True):
    """
    Consider the convex minimization problem
        f_* = min_x f(x),
    where f is L-smooth and mu-strongly convex
    .
    This code computes a worst-case guarantee for an inexact gradient method.
    That is, it computes the smallest possible tau(n,L,mu,epsilon) such that the guarantee
        f(x_n) - f_* <= tau(n,L,mu,epsilon) * ( f(x_0) - f_* )
    is valid, where x_n is the output of the gradient descent with an inexact descent direction,
    and where x_* is the minimizer of f.

    The inexact descent direction is assumed to satisfy a relative inaccuracy
    described by (with 0 <= epsilon < 1 )
        || f'(x_i) - d || <= epsilon * || f'(x_i) ||,
    where f'(x_i) is the true gradient, and d is the approximate descent direction that is used.

    The detailed approach (based on convex relaxations) is available in
    [1] De Klerk, Etienne, François Glineur, and Adrien B. Taylor.
    "On the worst-case complexity of the gradient method with exact line search for smooth strongly convex functions."
    Optimization Letters (2017).

    The tight guarantee obtained in [1, Theorem 5.1] is tau(n,L,mu,epsilon) = ((L_eps-mu_eps)/(L_eps+mu_eps))**(2*n),
    with L_eps = (1+epsilon) * L and mu_eps = (1-epsilon) * mu

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param epsilon: (float) level of inaccuracy
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
    # as well as corresponding inexact gradient and function value g0 and f0
    x0 = problem.set_initial_point()
    d0, f0 = inexactgradient(x0, func, epsilon, notion='relative')

    # Set the initial constraint that is the distance between f0 and f_*
    problem.set_initial_condition(f0 - fs <= 1)

    # Run n steps of the inexact gradient method
    Leps = (1 + epsilon) * L
    meps = (1 - epsilon) * mu
    gamma = 2 / (Leps + meps)
    x = x0
    dx = d0
    for i in range(n):
        x = x - gamma * dx
        dx, fx = inexactgradient(x, func, epsilon, notion='relative')

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = ((Leps - meps) / (Leps + meps)) ** (2 * n)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of inexact gradient ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 2
    L = 1
    mu = .1
    epsilon = .1

    pepit_tau, theoretical_tau = wc_InexactGrad(L=L,
                                                mu=mu,
                                                epsilon=epsilon,
                                                n=n)
