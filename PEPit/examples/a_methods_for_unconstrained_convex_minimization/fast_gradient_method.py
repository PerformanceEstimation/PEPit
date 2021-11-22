from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_fgm(mu, L, n, verbose=True):
    """
    Consider the convex minimization problem

    .. math:: f_* = \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex (:math:`\\mu` is possibly 0).

    This code computes a worst-case guarantee for the **fast gradient** method, a.k.a. **accelerated gradient** method.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu)` such that the guarantee

    .. math:: f(x_n) - f_* \\leqslant \\tau(n, L, \\mu) \\|x_0 - x_*\\|^2

    is valid, where :math:`x_n` is the output of the **accelerated gradient** method,
    and where :math:`x_*` is the minimizer of :math:`f`.
    In short, for given values of :math:`n`, :math:`L` and :math:`\\mu`,
    :math:`\\tau(n, L, \\mu)` is computed as the worst-case value of
    :math:`f(x_n)-f_*` when :math:`\\|x_0 - x_*\\|^2 \\leqslant 1`.

    **Algorithm**:

        .. math::
            :nowrap:

            \\begin{eqnarray}
                y_t & = & x_t + \\frac{t}{t + 3}(x_t - x_{t-1}) \\\\
                x_{t+1} & = & y_t - \\frac{1}{L} \\nabla f(y_t)
            \\end{eqnarray}

    **Theoretical guarantee**:

        The **upper** guarantee obtained in [1, 2] is

        .. math:: \\tau(n, L, \\mu) = \\frac{2L}{n^2 + 5n + 6}

        Furthermore, it is **tight** for smooth convex functions.

    References:

        Theoretical rates can be found in the following paper
        For an Upper bound (not tight):
        [1] A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems∗
        Amir Beck and Marc Teboulle

        For an exact bound (convex):
        [2] Exact Worst-case Performance of First-order Methods for Composite Convex Optimization
        Adrien B. Taylor, Julien M. Hendrickx, François Glineur

    Args:
        mu (float): the strong convexity parameter
        L (float): the smoothness parameter.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_fgm(mu=0, L=1, n=1, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 4x4
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 6 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.16666666668209376
        *** Example file: worst-case performance of conjugate gradient method ***
            PEP-it guarantee:		 f(x_n)-f_* <= 0.166667 ||x_0 - x_*||^2
            Theoretical guarantee:	 f(x_n)-f_* <= 0.166667 ||x_0 - x_*||^2

    """
    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the fast gradient method
    x_new = x0
    y = x0
    for i in range(n):
        x_old = x_new
        x_new = y - 1 / L * func.gradient(y)
        y = x_new + i / (i + 3) * (x_new - x_old)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func.value(x_new) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Theoretical guarantee (for comparison)
    theoretical_tau = 2 * L / (n ** 2 + 5 * n + 6)  # tight only for mu=0, see [2], Table 1 (column 1, line 1)
    if mu != 0:
        print('Warning: momentum is tuned for non-strongly convex functions.')

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of conjugate gradient method ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_fgm(mu=0, L=1, n=1, verbose=True)
