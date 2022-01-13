from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.primitive_steps import inexact_gradient_step


def wc_inexact_gradient(L, mu, epsilon, n, verbose=True):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for an **inexact gradient method**.
    That is, it computes the smallest possible :math:`\\tau(n,L,\\mu,\\varepsilon)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n,L,\\mu,\\varepsilon) (f(x_0) - f_\\star)

    is valid, where :math:`x_n` is the output of the gradient descent with an inexact descent direction,
    and where :math:`x_\\star` is the minimizer of :math:`f`.

    The inexact descent direction is assumed to satisfy a relative inaccuracy
    described by (with :math:`0 \\leqslant \\varepsilon \\leqslant 1`)

    .. math:: \|\\nabla f(x_t) - d_t\| \\leqslant \\varepsilon \\|\\nabla f(x_t)\\|,

    where :math:`\\nabla f(x_t)` is the true gradient, and :math:`d_t` is the approximate descent direction that is used.

    **Algorithm**:

    The inexact gradient descent under consideration can be written as

        .. math:: x_{t+1} = x_t - \\frac{2}{L_{\\varepsilon} + \\mu_{\\varepsilon}} d_t

    where :math:`d_t` is the inexact search direction, :math:`L_{\\varepsilon} = (1 + \\varepsilon)L`
    and :math:`\mu_{\\varepsilon} = (1-\\varepsilon) \\mu`.

    **Theoretical guarantee**:

    A **tight** worst-case guarantee obtained in [1, Theorem 5.3] or [2, Remark 1.6] is

        .. math:: f(x_n) - f_\\star \\leqslant \\left(\\frac{L_{\\varepsilon} - \\mu_{\\varepsilon}}{L_{\\varepsilon} + \\mu_{\\varepsilon}}\\right)^{2n}(f(x_0) - f_\\star ),

    with :math:`L_{\\varepsilon} = (1 + \\varepsilon)L` and :math:`\mu_{\\varepsilon} = (1-\\varepsilon) \\mu`. This
    guarantee is achieved on one-dimensional quadratic functions.

    **References**:The detailed analyses can be found in [1, 2].

    `[1] E. De Klerk, F. Glineur, A. Taylor (2020). Worst-case convergence analysis of
    inexact gradient and Newton methods through semidefinite programming performance estimation.
    SIAM Journal on Optimization, 30(3), 2053-2082.
    <https://arxiv.org/pdf/1709.05191.pdf>`_

    `[2] O. Gannot (2021). A frequency-domain analysis of inexact gradient methods.
    Mathematical Programming (to appear).
    <https://arxiv.org/pdf/1912.13494.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        epsilon (float): level of inaccuracy
        n (int): number of iterations.
        verbose (bool): if True, print conclusion

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_inexact_gradient(L=1, mu=0.1, epsilon=0.1, n=2, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 8x8
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 15 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.5188661397616067
        (PEPit) Postprocessing: applying trace heuristic. Currently 4 eigenvalue(s) > 1e-05 before resolve.
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); objective value: 0.5188511226295036
        (PEPit) Postprocessing: 3 eigenvalue(s) > 1e-05 after trace heuristic
        *** Example file: worst-case performance of inexact gradient ***
            PEPit guarantee:		 f(x_n)-f_* <= 0.518851 (f(x_0)-f_*)
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

    # Set the initial constraint that is the distance between f0 and f_*
    problem.set_initial_condition(func.value(x0) - fs <= 1)

    # Run n steps of the inexact gradient method
    Leps = (1 + epsilon) * L
    meps = (1 - epsilon) * mu
    gamma = 2 / (Leps + meps)

    x = x0
    for i in range(n):
        x, dx, fx = inexact_gradient_step(x, func, gamma=gamma, epsilon=epsilon, notion='relative')

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func.value(x) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose, tracetrick=True)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = ((Leps - meps) / (Leps + meps)) ** (2 * n)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of inexact gradient ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_inexact_gradient(L=1, mu=0.1, epsilon=0.1, n=2, verbose=True)
