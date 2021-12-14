import numpy as np

from PEPit.pep import PEP
from PEPit.functions.convex_function import ConvexFunction
from PEPit.functions.convex_indicator import ConvexIndicatorFunction
from PEPit.primitive_steps.bregman_gradient_step import bregman_gradient_step


def wc_no_lips_1(L, gamma, n, verbose=True):
    """
    Consider the constrainted non-convex minimization problem

    .. math:: F_\\star \\triangleq \\min_x \\{F(x) \equiv f_1(x)+f_2(x) \\}

    where :math:`f_2` is a closed convex indicator function and :math:`f_1` is :math:`L`-smooth relatively to :math:`h` (possibly non-convex),
    and :math:`h` is closed proper and convex.

    This code computes a worst-case guarantee for the **NoLips** method solving this problem.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\gamma)` such that the guarantee

    .. math:: \\min_{0 \\leqslant t \\leqslant n-1} D_h(x_{t+1}, x_t) \\leqslant \\tau(n, L, \\gamma)  (F(x_0) - F(x_n))

    is valid, where :math:`x_n` is the output of the **NoLips** method,
    and where :math:`D_h` is the Bregman distance generated by :math:`h`:

    .. math:: D_h(x, y) \\triangleq h(x) - h(y) - \\nabla h (y)^T(x - y).

    **Algorithms**:

    For :math:`t \\in \\{0, \\dots, n-1\\}`,

    .. math:: x_{t+1} = \\arg\\min_{u \\in R^d} \\nabla f(x_t)^T(u - x_t) + \\frac{1}{\\gamma} D_h(u, x_t).

    **Theoretical guarantees**:

    The **tight** theoretical upper bound is obtained in [1, Proposition 4.1]

        .. math:: \\min_{0 \\leqslant t \\leqslant n-1} D_h(x_{t+1}, x_t) \\leqslant \\frac{\\gamma}{n(1 - L\\gamma)}(F(x_0) - F(x_n))

    **References**:

    The detailed approach is availaible in [1], and the PEP approach is presented in [2].

    `[1] J. Bolte, S. Sabach, M. Teboulle, and Y. Vaisbourd (2017). First Order Methods Beyond
    Convexity and Lipschitz Gradient Continuity with Applications to Quadratic Inverse Problems (SIAM
    Journal on Optimization).
    <https://arxiv.org/pdf/1706.06461.pdf>`_

    `[2] R.-A. Dragomir, A. B. Taylor, A. d’Aspremont, and
    J. Bolte (2019). Optimal Complexity and Certification of Bregman
    First-Order Methods (Mathematical Programming).
    <https://arxiv.org/pdf/1911.08510.pdf>`_

    DISCLAIMER: This example requires some experience with PESTO and PEPs
    (see Section 4 in [2]).

    Args:
        L (float): relative-smoothness parameter.
        gamma (float): step-size (equal to 1/(2*L) for guarantee).
        n (int): number of iterations.
        verbose (bool, optional): if True, print conclusion.

    Returns:
        tuple: worst-case value, theoretical value

    Example:
        >>> L = 1
        >>> pepit_tau, theoretical_tau = wc_no_lips_1(L=L, gamma=1 / (2 * L), n=5, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 20x20
        (PEP-it) Setting up the problem: performance measure is minimum of 5 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 3 function(s)
                 function 1 : 132 constraint(s) added
                 function 2 : 110 constraint(s) added
                 function 3 : 49 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.2000008155245043
        *** Example file: worst-case performance of the NoLips in function values ***
            PEP-it guarantee:		 min_t Dh(x_(t+1)), x_(t)) <= 0.200001 (F(x_0) - F(x_n))
            Theoretical guarantee :	 min_t Dh(x_(t+1), x_(t)) <= 0.2 (F(x_0) - F(x_n))

    """

    # Instantiate PEP
    problem = PEP()

    # Declare two convex functions and a convex indicator function
    d1 = problem.declare_function(ConvexFunction, param={}, is_differentiable=True)
    d2 = problem.declare_function(ConvexFunction, param={}, is_differentiable=True)
    func1 = (d2 - d1) / 2
    h = (d1 + d2) / 2 / L
    func2 = problem.declare_function(ConvexIndicatorFunction, param={'D': np.inf})

    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()
    gh0, h0 = h.oracle(x0)
    gf0, f0 = func1.oracle(x0)
    _, F0 = func.oracle(x0)

    # Compute n steps of the NoLips starting from x0
    xx = [x0 for _ in range(n + 1)]
    gfx = gf0
    ghx = [gh0 for _ in range(n + 1)]
    hx = [h0 for _ in range(n + 1)]
    for i in range(n):
        xx[i + 1], _, _ = bregman_gradient_step(gfx, ghx[i], func2 + h, gamma)
        gfx, _ = func1.oracle(xx[i + 1])
        ghx[i + 1], hx[i + 1] = h.oracle(xx[i + 1])
        Dh = hx[i + 1] - hx[i] - ghx[i] * (xx[i + 1] - xx[i])
        # Set the performance metric to the final distance in Bregman distances to the last iterate
        problem.set_performance_metric(Dh)
    _, Fx = func.oracle(xx[n])

    # Set the initial constraint that is the distance in function values between x0 and x^*
    problem.set_initial_condition(F0 - Fx <= 1)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = gamma / (n * (1 - L * gamma))

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the NoLips in function values ***')
        print('\tPEP-it guarantee:\t\t min_t Dh(x_(t+1)), x_(t)) <= {:.6} (F(x_0) - F(x_n))'.format(pepit_tau))
        print('\tTheoretical guarantee :\t min_t Dh(x_(t+1), x_(t)) <= {:.6} (F(x_0) - F(x_n))'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    L = 1
    pepit_tau, theoretical_tau = wc_no_lips_1(L=L, gamma=1 / (2 * L), n=5, verbose=True)