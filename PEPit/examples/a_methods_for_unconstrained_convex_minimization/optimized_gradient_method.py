from math import sqrt

from PEPit.pep import PEP
from PEPit.functions.smooth_convex_function import SmoothConvexFunction


def wc_ogm(L, n, verbose=True):
    """
    Consider the minimization problem

    .. math:: f_\star = \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code computes a worst-case guarantee for **optimized gradient method** (OGM). That is, it computes
    the smallest possible :math:`\\tau(n, L)` such that the guarantee

    .. math:: f(x_n) - f_\star \\leqslant \\tau(n, L)  || x_0 - x_\star ||^2

    is valid, where :math:`x_n` is the output of OGM and where :math:`x_\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n` and :math:`L`, :math:`\\tau(n, L)` is computed as the worst-case value
    of :math:`f(x_n)-f_\star` when :math:`||x_0 - x_\star||^2 \\leqslant 1`.

    **Algorithm**:
    The optimized gradient method is described by

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_{t+1} & = & y_t - \\frac{1}{L} \\nabla f(y_t)\\\\
                y_{k+1} & = & x_{t+1} + \\frac{\\theta_{t}-1}{\\theta_{t+1}}(x_{t+1}-x_t)+\\frac{\\theta_{t}}{\\theta_{t+1}}(x_{t+1}-y_t),
            \\end{eqnarray}

    with

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\theta_0 & = & 1 \\\\
                \\theta_t & = & \\frac{1 + \\sqrt{4 \\theta_{t-1}^2 + 1}}{2}, \\forall t \\in [|1, n-1|] \\\\
                \\theta_n & = & \\frac{1 + \\sqrt{8 \\theta_{n-1}^2 + 1}}{2}.
            \\end{eqnarray}

    **Theoretical guarantee**:
    The **tight** theoretical guarantee can be found in [2, Theorem 2]:

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{L||x_0-x_\\star||^2}{2\\theta_n^2}.

    **References**:
    The optimized gradient method was developed in the following two works:
    [1] Y. Drori, M. Teboulle (2014).Performance of first-order methods for smooth convex minimization: a novel
    approach. Mathematical Programming 145.1-2: 451-482.

    [2] D. Kim, J. Fessler (2016).Optimized first-order methods for smooth convex minimization. Mathematical
    programming 159.1-2: 81-107.

    Args:
        L (float): the smoothness parameter.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_ogm(L=3, n=4, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 7x7
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 30 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: MOSEK); optimal value: 0.07675182659831646
        *** Example file: worst-case performance of optimized gradient method ***
	        PEP-it guarantee:       f(y_n)-f_* <= 0.0767518 || x_0 - x_* ||^2
	        Theoretical guarantee:  f(y_n)-f_* <= 0.0767518 || x_0 - x_* ||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction, param={'mu': 0, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the optimized gradient method (OGM) method
    theta_new = 1
    x_new = x0
    y = x0
    for i in range(n):
        x_old = x_new
        x_new = y - 1 / L * func.gradient(y)
        theta_old = theta_new
        if i < n - 1:
            theta_new = (1 + sqrt(4 * theta_new ** 2 + 1)) / 2
        else:
            theta_new = (1 + sqrt(8 * theta_new ** 2 + 1)) / 2

        y = x_new + (theta_old - 1) / theta_new * (x_new - x_old) + theta_old / theta_new * (x_new - y)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func.value(y) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / (2 * theta_new ** 2)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of optimized gradient method ***')
        print('\tPEP-it guarantee:\t\t f(y_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(y_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_ogm(L=3, n=4, verbose=True)
