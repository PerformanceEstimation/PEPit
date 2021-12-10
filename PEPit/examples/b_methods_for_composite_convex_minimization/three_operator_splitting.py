from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.functions.smooth_convex_function import SmoothConvexFunction
from PEPit.functions.convex_function import ConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_tos(mu1, L1, L3, alpha, theta, n, verbose=True):
    """
    Consider the composite convex minimization problem,

    .. math:: \\min_x {F(x) \\equiv f_1(x) + f_2(x) + f_3(x)}

    where,

    :math:`f_1` is :math:`L_1`-smooth and :math:`\\mu_1`-strongly convex,
    :math:`f_2` is closed, convex and proper,
    and :math:`f_3` is :math:`L_3`-smooth convex.
    Proximal operators are assumed to be available for :math:`f_1` and :math:`f_2`.

    This code computes a worst-case guarantee for the **Three Operator Splitting (TOS)**.
    That is, it computes the smallest possible :math:`\\tau(n, L_1, L_3, \\mu_1)` such that the guarantee

    .. math:: \|w^{(0)}_{n} - w^{(1)}_{n}\|^2 \\leqslant \\tau(n, L_1, L_3, \\mu_1, \\alpha, \\theta) \|w^{(0)}_{0} - w^{(1)}_{0}\|^2

    is valid, where :math:`w^{(0)}_{0}` and :math:`w^{(1)}_{0}` are two different starting points
    and :math:`w^{(0)}_{n}` and :math:`w^{(1)}_{n}` are the two corresponding :math:`n^{\\mathrm{th}}` outputs of TOS.
    (i.e., how do the iterates contract when the method is started from two different initial points).

    In short, for given values of :math:`n`, :math:`L_1`, :math:`L_3`, :math:`\\mu_1`, :math:`\\alpha` and :math:`\\theta`,
    the contraction factor :math:`\\tau(n, L_1, L_3, \\mu_1, \\alpha, \\theta)` is computed as the worst-case value of
    :math:`\\|w^{(0)}_{n} - w^{(1)}_{n}\\|^2` when :math:`\\|w^{(0)}_{0} - w^{(1)}_{0}\\|^2 \\leqslant 1`.

    **Algorithm**:
    One iteration of the algorithm (see [1]) is described by

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_k & = & \\mathrm{prox}_{\\alpha, f_2}(w_k) \\\\
                y_k & = & \\mathrm{prox}_{\\alpha, f_1}(2 x_k - w_k - \\alpha \\nabla f_3(x_k)) \\\\
                w_{k+1} & = & w_k + \\theta (y_k - x_k)
            \\end{eqnarray}

    References:

        The TOS is introduced in [1].

        `[1] Damek Davis, and Wotao Yin. "A three-operator splitting scheme and its optimization applications."
        Set-valued and variational analysis  (2017).
        <https://arxiv.org/pdf/1504.01032.pdf>`_

    Args:
        mu1 (float): the strong convexity parameter.
        L1 (float): the smoothness parameter of function f1.
        L3 (float): the smoothness parameter of function f3.
        alpha (float): parameter of the scheme.
        theta (float): parameter of the scheme.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> L3 = 1
        >>> pepit_tau, theoretical_tau = wc_tos(mu1=0.1, L1=10, L3=L3, alpha=1 / L3, theta=1, n=4, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 29x29
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 3 function(s)
                 function 1 : 72 constraint(s) added
                 function 2 : 72 constraint(s) added
                 function 3 : 72 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.4752811057240984
        *** Example file: worst-case performance of the Three Operator Splitting in distance ***
            PEP-it guarantee:		 ||w^2_n - w^1_n||^2 <= 0.475281 ||x0 - ws||^2
            Theoretical guarantee :	 ||w^2_n - w^1_n||^2 <= 0.5 ||x0 - ws||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function, a smooth function and a convex function.
    func1 = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu1, 'L': L1})
    func2 = problem.declare_function(ConvexFunction, param={})
    func3 = problem.declare_function(SmoothConvexFunction, param={'L': L3})
    # Define the function to optimize as the sum of func1, func2 and func3
    func = func1 + func2 + func3

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()

    # Then define the starting points w0 and w0p of the algorithm
    w0 = problem.set_initial_point()
    w0p = problem.set_initial_point()

    # Set the initial constraint that is the distance between w0 and w0p
    problem.set_initial_condition((w0 - w0p) ** 2 <= 1)

    # Compute n steps of the Three Operator Splitting starting from w0
    w = w0
    for _ in range(n):
        x, _, _ = proximal_step(w, func2, alpha)
        gx, _ = func3.oracle(x)
        y, _, _ = proximal_step(2 * x - w - alpha * gx, func1, alpha)
        w = w + theta * (y - x)

    # Compute trajectory starting from w0p
    wp = w0p
    for _ in range(n):
        xp, _, _ = proximal_step(wp, func2, alpha)
        gxp, _ = func3.oracle(xp)
        yp, _, _ = proximal_step(2 * xp - wp - alpha * gxp, func1, alpha)
        wp = wp + theta * (yp - xp)

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric((w - wp) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = None

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Three Operator Splitting in distance ***')
        print('\tPEP-it guarantee:\t\t ||w^2_n - w^1_n||^2 <= {:.6} ||x0 - ws||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    L3 = 1
    pepit_tau, theoretical_tau = wc_tos(mu1=0.1, L1=10, L3=L3, alpha=1 / L3, theta=1, n=4, verbose=True)
