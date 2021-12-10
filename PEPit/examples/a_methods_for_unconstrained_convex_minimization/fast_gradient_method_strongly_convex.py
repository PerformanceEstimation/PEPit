from math import sqrt

from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_fgm(mu, L, n, verbose=True):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for the **fast gradient** method, a.k.a **accelerated gradient** method.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L, \\mu) \\left(f(x_0) -  f(x_\\star) + \\frac{\\mu}{2}\\|x_0 - x_\\star\\|^2\\right),

    is valid, where :math:`x_n` is the output of the **accelerated gradient** method,
    and where :math:`x_\\star` is the minimizer of :math:`f`.
    In short, for given values of :math:`n`, :math:`L` and :math:`\\mu`,
    :math:`\\tau(n, L, \\mu)` is computed as the worst-case value of
    :math:`f(x_n)-f_\\star` when :math:`f(x_0) -  f(x_\\star) + \\frac{\\mu}{2}\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    For :math:`t \\in \\{0, \\dots, n-1\\}`,

        .. math::
            :nowrap:

            \\begin{eqnarray}
                y_t & = & x_t + \\frac{\\sqrt{L} - \\sqrt{\\mu}}{\\sqrt{L} + \\sqrt{\\mu}}(x_t - x_{t-1}) \\\\
                x_{t+1} & = & y_t - \\frac{1}{L} \\nabla f(y_t)
            \\end{eqnarray}

    with :math:`x_{-1}:= x_0`.

    **Theoretical guarantee**:

        The **upper** guarantee obtained in [1,  Corollary 4.15] is

        .. math:: f(x_n)-f_\\star \\leqslant \\left(1 - \\sqrt{\\frac{\\mu}{L}}\\right)^n \\left(f(x_0) -  f(x_\\star) + \\frac{\\mu}{2}\\|x_0 - x_\\star\\|^2\\right).

    References:
        A theoretical **upper** bound can be found in [1,  Corollary 4.15].

        `[1] A. d’Aspremont, D. Scieur, A. Taylor, (2021),  Acceleration Methods, Monograph.
        <https://arxiv.org/pdf/2101.09545.pdf>`_

    Args:
        mu (float): the strong convexity parameter
        L (float): the smoothness parameter.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_fgm(mu=0.1, L=1, n=2, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 5x5
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 12 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.34758587217463155
        *** Example file: worst-case performance of the accelerated gradient method ***
            PEP-it guarantee:		 f(x_n)-f_*  <= 0.347586 (f(x_0) -  f(x_*) +  mu/2*|| x_0 - x_* ||**2)
            Theoretical guarantee:	 f(x_n)-f_*  <= 0.467544 (f(x_0) -  f(x_*) +  mu/2*|| x_0 - x_* ||**2)

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

    # Set the initial constraint that is a well-chosen distance between x0 and x^*
    problem.set_initial_condition(func.value(x0) - fs + mu / 2 * (x0 - xs) ** 2 <= 1)

    # Run n steps of the fast gradient method
    kappa = mu / L
    x_new = x0
    y = x0
    for i in range(n):
        x_old = x_new
        x_new = y - 1 / L * func.gradient(y)
        y = x_new + (1 - sqrt(kappa)) / (1 + sqrt(kappa)) * (x_new - x_old)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func.value(x_new) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (1 - sqrt(kappa)) ** n
    if mu == 0:
        print("Warning: momentum is tuned for strongly convex functions!")

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the accelerated gradient method ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_*  <= {:.6} (f(x_0) -  f(x_*) +  mu/2*|| x_0 - x_* ||**2)'.format(
            pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_*  <= {:.6} (f(x_0) -  f(x_*) +  mu/2*|| x_0 - x_* ||**2)'.format(
            theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_fgm(mu=0.1, L=1, n=2, verbose=True)
