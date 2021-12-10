from PEPit.pep import PEP
from PEPit.functions.smooth_convex_function import SmoothConvexFunction
from PEPit.primitive_steps.inexact_gradient import inexact_gradient


def wc_InexactAGM(L, epsilon, n, verbose=True):
    """
    Consider the minimization problem

    .. math:: f_\star = \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code computes a worst-case guarantee for an **accelerated gradient method** using **inexact first-order
    information**. That is, it computes the smallest possible :math:`\\tau(n, L, \\epsilon)` such that the guarantee

    .. math:: f(x_n) - f_\star \\leqslant \\tau(n, L, \\epsilon)  || x_0 - x_\star ||^2

    is valid, where :math:`x_n` is the output of **inexact accelerated gradient descent** and where :math:`x_\star`
    is a minimizer of :math:`f`.

    The inexact descent direction is assumed to satisfy a relative inaccuracy described by
    (with :math:`0\\leqslant \\epsilon \\leqslant 1`)

    .. math:: || f'(y_i) - d_i || \\leqslant \\epsilon  || f'(y_i) ||,

    where :math:`f'(y_i)` is the true gradient at :math:`y_i` and :math:`d_i` is the approximate descent direction that is used.

    **Algorithm**:
    The inexact accelerated gradient method of this example is provided by

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_{k+1} &&= y_k - \\frac{1}{L} d_k\\\\
                y_{k+1} &&= x_{k+1} + \\frac{k-1}{k + 2}  (x_{k+1} - x_k).
            \\end{eqnarray}

    **Theoretical guarantee**:
    When :math:`\\epsilon=0`, a **tight** theoretical guarantee can be found in [1, Table 1]:

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{2L||x_0-x_\\star||^2}{n^2 + 5 n + 6}.

    **References**:
    [1] A. Taylor, J. Hendrickx, F. Glineur (2017). Exact worst-case performance of first-order methods for composite
    convex optimization. SIAM Journal on Optimization, 27(3):1283–1313.

    Args:
        L (float): smoothness parameter.
        epsilon (float): level of inaccuracy
        n (int): number of iterations.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_InexactAGM(L=3, epsilon=.1, n=5, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 13x13
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
		         function 1 : 47 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: MOSEK); optimal value: 0.11816352677249196
        *** Example file: worst-case performance of inexact accelerated gradient method ***
	        PEP-it guarantee:                       f(x_n)-f_* <= 0.118164 (f(x_0)-f_*)
	        Theoretical guarantee for epsilon = 0:  f(x_n)-f_* <= 0.107143 (f(x_0)-f_*)
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothConvexFunction, param={'mu': 0, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the inexact accelerated gradient method
    x_new = x0
    y = x0
    for i in range(n):
        dy, fy = inexact_gradient(y, func, epsilon, notion='relative')
        x_old = x_new
        x_new = y - 1 / L * dy
        y = x_new + i / (i + 3) * (x_new - x_old)
    _, fx = func.oracle(x_new)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 2 * L / (n ** 2 + 5 * n + 6)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of inexact accelerated gradient method ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee for epsilon = 0 :\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 5
    L = 1
    epsilon = 0.1  # Theoretical and PEPit guarantee match when epsilon = 0.

    pepit_tau, theoretical_tau = wc_InexactAGM(L=L,
                                               epsilon=epsilon,
                                               n=n)
