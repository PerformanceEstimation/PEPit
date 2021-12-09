import numpy as np

from PEPit.pep import PEP
from PEPit.functions.convex_lipschitz_function import ConvexLipschitzFunction


def wc_subgd(M, N, gamma, verbose=True):
    """
    Consider the minimization problem

    .. math:: f_\star = \\min_x f(x),

    where :math:`f` is convex and :math:`M`-Lipschitz. This problem is a (possibly non-smooth) minimization problem.

    This code computes a worst-case guarantee for the **subgradient** method. That is, it computes
    the smallest possible :math:`\\tau(n, M)` such that the guarantee

    .. math:: \\min_{0 \leqslant t \leqslant N} f(x_t) - f_\star \\leqslant \\tau(n, M)  \|x_0 - x_\star\|^2

    is valid, where :math:`x_i` is the output of the **subgradient** method,
    and where :math:`x_\star` is the minimizer of :math:`f`.

    We show how to compute the worst-case value of :math:`\\min_t F(x_t)-F(x_\star)` when :math:`x_t` is
    obtained by doing :math:`i` steps of a subgradient method starting with an initial
    iterate satisfying :math:`\\|x_0-x_\\star\\| \\leqslant 1`.

    **Algorithm**:

        .. math::
            :nowrap:

            \\begin{eqnarray}
                g_{t} & \\in & \\partial f(x_t) \\\\
                x_{t+1} & = & x_t - \\gamma g_t
            \\end{eqnarray}

    **Theoretical guarantee**:

    The **tight** bound is obtained in [1, Section 3.2.3],

        .. math:: \\min_{0 \\leqslant t \\leqslant n} F(x_t)-F(x_\star) \\leqslant \\frac{M}{\\sqrt{n+1}}\|x_0-x_\star\|.

    **References**:

        `[1] Y. Nesterov, Introductory Lectures on Convex Programming, Volume 1: Basic course (1998).
        <https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=DJ8Ep8YAAAAJ&citation_for_view=DJ8Ep8YAAAAJ:FiDNX6EVdGUC>`_

    Args:
        M (float): the Lipschitz parameter.
        N (int): the number of iterations.
        gamma (float): step size.
        verbose (bool, optional): if True, print conclusion.

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> M = 2
        >>> N = 6
        >>> gamma = 1 / (M * np.sqrt(N + 1))
        >>> pepit_tau, theoretical_tau = wc_subgd(M, N, gamma, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 9x9
        (PEP-it) Setting up the problem: performance measure is minimum of 7 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 64 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.755982533173183
        *** Example file: worst-case performance of subgradient method ***
            PEP-it guarantee:		 min_(0 <= i <= N) f(x_i) - f_*  <= 0.755983 ||x_0 - x_*||**2`
            Theoretical guarantee:	 min_(0 <= i <= N) f(x_i) - f_*  <= 0.755929 ||x_0 - x_*||**2`
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex lipschitz function
    func = problem.declare_function(ConvexLipschitzFunction,
                                    param={'M': M})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the subgradient method
    x = x0
    gx, fx = func.oracle(x)

    for _ in range(N):
        problem.set_performance_metric(fx - fs)
        x = x - gamma * gx
        gx, fx = func.oracle(x)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = M / np.sqrt(N + 1)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of subgradient method ***')
        print('\tPEP-it guarantee:\t\t min_(0 \leq i \leq N) f(x_i) - f_*  <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t min_(0 \leq i \leq N) f(x_i) - f_*  <= {:.6} ||x_0 - x_*||^2'.format(
            theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    M = 2
    N = 6
    gamma = 1 / (M * np.sqrt(N + 1))

    rate = wc_subgd(M=M, N=N, gamma=gamma, verbose=True)
