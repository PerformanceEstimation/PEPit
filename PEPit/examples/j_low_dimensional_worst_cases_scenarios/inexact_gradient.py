from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.primitive_steps.inexact_gradient import inexact_gradient


def wc_InexactGrad(L, mu, epsilon, n, verbose=True):
    """
    Consider the convex minimization problem

        .. math:: f_\\star = \min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for an **inexact gradient method**.
    That is, it computes the smallest possible :math:`\\tau(n,L,\\mu,\\epsilon)` such that the guarantee

        .. math:: f(x_n) - f_\\star \\leqslant \\tau(n,L,\\mu,\\epsilon) ( f(x_0) - f_\\star )
    is valid, where :math:`x_n` is the output of the gradient descent with an inexact descent direction,
    and where :math:`x_\\star` is the minimizer of :math:`f`.

    The inexact descent direction is assumed to satisfy a relative inaccuracy
    described by (with :math:`0 \\leqslant \\epsilon \\leqslant 1` )

        .. math:: || \\nabla f(x_t) - d_t || \\leqslant \\epsilon || \\nabla f(x_t) ||,

    where :math:`\\nabla f(x_t)` is the true gradient, and :math:`d_t` is the approximate descent direction that is used.

    **Algorithm**:

    The inexact gradient descent under consideration can be written as

        .. math:: x_{t+1} = x_t - \\frac{2}{L_{\\epsilon} + \\mu_{\\epsilon}} d_t

    where :math:`d_t` is the inexact search direction, :math:`L_{\\epsilon} = (1 + \\epsilon)L`
    and :math:`\mu_{\\epsilon} = (1-\\epsilon) \\mu`.

    **Theoretical guarantee**:

    The **tight** worst-case guarantee obtained in [1, Theorem 5.3] or [2, Remark 1.6] is

        .. math:: f(x_n) - f_\\star \\leqslant \\left(\\frac{L_{\\epsilon} - \\mu_{\\epsilon}}{L_{\\epsilon} + \\mu_{\\epsilon}}\\right)^{2n}(f(x_0) - f_\\star ),

    with :math:`L_{\\epsilon} = (1 + \\epsilon)L` and :math:`\mu_{\\epsilon} = (1-\\epsilon) \\mu`.

    **References**:

        The detailed approach and proof are available in [1, 2].

        [1] E. De Klerk, F. Glineur, A. Taylor (2020). Worst-case convergence analysis of
        inexact gradient andNewton methods through semidefinite programming performance estimation.
        SIAM Journal on Optimization, 30(3), 2053-2082.

        [2] O. Gannot (2021). A frequency-domain analysis of inexact gradient methods. Mathematical Programming.

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param epsilon: (float) level of inaccuracy
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_InexactGrad(L=1, mu=0.1, epsilon=0.1, n=2, Verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 8x8
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 15 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.5188606799005029
        (PEP-it) Postprocessing: applying trace heuristic. Currently 5 eigenvalue(s) > 1e-05 before resolve.
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); objective value: 0.5188668411169357
        (PEP-it) Postprocessing: 4 eigenvalue(s) > 1e-05 after trace heuristic
        *** Example file: worst-case performance of inexact gradient ***
            PEP-it guarantee:		 f(x_n)-f_* <= 0.518867 (f(x_0)-f_*)
            Theoretical guarantee:	 f(x_n)-f_* <= 0.518917 (f(x_0)-f_*)
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
    d0, f0 = inexact_gradient(x0, func, epsilon, notion='relative')

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
        dx, fx = inexact_gradient(x, func, epsilon, notion='relative')

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose, tracetrick=True)

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
