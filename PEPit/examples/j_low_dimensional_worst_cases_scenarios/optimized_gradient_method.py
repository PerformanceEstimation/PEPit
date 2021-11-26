from math import sqrt

from PEPit.pep import PEP
from PEPit.functions.smooth_convex_function import SmoothConvexFunction


def wc_ogm(L, n, verbose=True):
    """
    Consider the minimization problem

    .. math:: f_\star = \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code computes a worst-case guarantee for **optimized gradient method** (OGM), and applies the trace heuristic
    for trying to find a low-dimensional worst-case example on which this guarantee is achieved. That is, it computes
    the smallest possible :math:`\\tau(n, L)` such that the guarantee

    .. math:: f(x_n) - f_\star \\leqslant \\tau(n, L)  || x_0 - x_\star ||^2

    is valid, where :math:`x_n` is the output of OGM and where :math:`x_\star` is a minimizer of :math:`f`. Then,
    it applies the trace heuristic, which allows obtaining a one-dimensional function on which the guarantee is achieved.

    **Algorithm**:
    The optimized gradient method is described by

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_{k+1} &&= y_k - \\frac{1}{L} \\nabla f(y_k)\\\\
                y_{k+1} &&= x_{k+1} + \\frac{\\theta_{k}-1}{\\theta_{k+1}}(x_{k+1}-x_k)+\\frac{\\theta_{k}}{\\theta_{k+1}}(x_{k+1}-y_k),
            \\end{eqnarray}

    with

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\theta_0 & = & 1 \\\\
                \\theta_i & = & \\frac{1 + \\sqrt{4 \\theta_{i-1}^2 + 1}}{2}, \\forall i \\in [|1, n-1|] \\\\
                \\theta_n & = & \\frac{1 + \\sqrt{8 \\theta_{n-1}^2 + 1}}{2}.
            \\end{eqnarray}
    **Theoretical guarantee**:
    The tight theoretical guarantee can be found in [2, Theorem 2]:

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{L||x_0-x_\\star||^2}{2\\theta_n^2}.

    **References**:
    The OGM was developed in [1,2]. Low-dimensional worst-case functions for OGM were obtained in [3, 4]

    [1] Y. Drori, M. Teboulle (2014).Performance of first-order methods for smooth convex minimization: a novel
    approach. Mathematical Programming 145.1-2: 451-482.

    [2] D. Kim, J. Fessler (2016).Optimized first-order methods for smooth convex minimization. Mathematical
    programming 159.1-2: 81-107.

    [3] A. Taylor, J. Hendrickx, F. Glineur (2017). Smooth strongly convex interpolation and exact worst-case
    performance of first-order methods. Mathematical Programming, 161(1-2), 307-345.

    [4] D. Kim, J. Fessler (2017). On the convergence analysis of the optimized gradient method. Journal of
    Optimization Theory and Applications, 172(1), 187-205.

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
        (PEP-it) Postprocessing: applying trace heuristic. Currently 6 eigenvalue(s) > 1e-05 before resolve.
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: MOSEK); objective value: 0.07674182628815357
        (PEP-it) Postprocessing: 1 eigenvalue(s) > 1e-05 after trace heuristic
        *** Example file: worst-case performance of optimized gradient method ***
	        PEP-it guarantee:		 f(y_n)-f_* <= 0.0767418 || x_0 - x_* ||^2
	        Theoretical guarantee:	 f(y_n)-f_* <= 0.0767518 || x_0 - x_* ||^2
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
    pepit_tau = problem.solve(verbose=verbose, tracetrick=True)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / 2 / theta_new ** 2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of optimized gradient method ***')
        print('\tPEP-it guarantee:\t\t f(y_n)-f_* <= {:.6} || x_0 - x_* ||^2'.format(
            pepit_tau))
        print('\tTheoretical guarantee:\t f(y_n)-f_* <= {:.6} || x_0 - x_* ||^2'.format(
            theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 2
    L = 1

    pepit_tau, theoretical_tau = wc_ogm(L=L,
                                        n=n)
