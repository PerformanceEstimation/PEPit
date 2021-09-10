from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Primitive_steps.exactlinesearch_step import exactlinesearch_step
from PEPit.Primitive_steps.inexactgradient import inexactgradient


def wc_InexactGrad_ELS(L, mu, epsilon, n):
    """
    Consider the convex minimization problem
        f_* = min_x f(x),
    where f is L-smooth and mu-strongly convex.
    This code computes a worst-case guarantee for an inexact gradient method with exact linesearch (ELS).
    That is, it computes the smallest possible tau(n,L,mu) such that the guarantee
        f(x_n) - f_* <= tau(n,L,mu,epsilon) * ( f(x_0) - f_* )
    is valid, where x_n is the output of the gradient descent with an inexact descent direction and an exact linesearch,
    and where x_* is the minimizer of f. The inexact descent direction is assumed to satisfy a relative inaccuracy
    described by (with 0 <= epsilon <= 1 )
        || f'(x_i) - d || <= epsilon * || f'(x_i) ||,
    where f'(x_i) is the true gradient, and d is the approximate descent direction that is used.

    In short, for given values of n and L, tau(n,L) is be computed as the worst-case value of f(x_n)-f_* when
    f(x_0) - f_* == 1.

    The detailed approach (based on convex relaxations) is available in
    [1] De Klerk, Etienne, François Glineur, and Adrien B. Taylor.
    "On the worst-case complexity of the gradient method with exact line search for smooth strongly convex functions."
    Optimization Letters (2017).

    The tight guarantee obtained in [1] is tau(n,L,mu,epsilon) = ((L_eps-mu_eps)/(L_eps+mu_eps))**(2*n),
    with L_eps = (1+epsilon) * L and mu_eps = (1-epsilon) * mu

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param epsilon: (float) level of inaccuracy
    :param n: (int) number of iterations.
    :return:
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, {'mu': mu, 'L': L})

    # Start by defining its unique optimal point
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition( f0 - fs <= 1)

    # Run the GD method with ELS
    x = x0
    for i in range(n):
        dx, _ = inexactgradient(x, func, epsilon, notion='relative')
        x, gx, fx = exactlinesearch_step(x, func, [dx])

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    wc = problem.solve()
    # Theoretical guarantee (for comparison)
    Leps = (1+epsilon) * L
    meps = (1-epsilon) * mu
    theory = ((Leps-meps)/(Leps+meps))**(2*n)

    print('*** Example file: worst-case performance of inexact gradient descent with exact linesearch (ELS) ***')
    print('\tPEP-it guarantee:\t f(x_n)-f_* <= ', wc)
    print('\tTheoretical guarantee:\t f(x_n)-f_* <= ', theory)
    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return wc, theory


if __name__ == "__main__":
    n = 2
    L = 1
    mu= .1
    epsilon = .1

    wc,theory = wc_InexactGrad_ELS(L=L, mu=mu, epsilon=epsilon, n=n)

    print('{}'.format(wc))
    print('{}'.format(theory))
