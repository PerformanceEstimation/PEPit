from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.functions import ConvexFunction
from PEPit.primitive_steps import proximal_step


def wc_douglas_rachford_splitting_contraction(mu, L, alpha, theta, n, verbose=True):
    """
    Consider the composite convex minimization problem

        .. math:: F_\\star \\triangleq \min_x \\{F(x) \equiv f_1(x) + f_2(x) \\}

    where :math:`f_1(x)` is :math:`L`-smooth and :math:`\mu`-strongly convex, and :math:`f_2` is convex,
    closed and proper. Both proximal operators are assumed to be available.

    This code computes a worst-case guarantee for the **Douglas Rachford Splitting (DRS)** method.
    That is, it computes the smallest possible :math:`\\tau(\\mu,L,\\alpha,\\theta,n)` such that the guarantee

        .. math:: \|w_1 - w_1'\|^2 \\leqslant \\tau(\\mu,L,\\alpha,\\theta,n) \|w_0 - w_0'\|^2.

    is valid, where :math:`x_n` is the output of the **Douglas Rachford Splitting method**. It is a contraction
    factor computed when the algorithm is started from two different points :math:`w_0` and :math:`w_0`.

    **Algorithm**:

    Our notations for the DRS method are as follows [3, Section 7.3], for :math:`t \\in \\{0, \\dots, n-1\\}`,

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_t & = & \\mathrm{prox}_{\\alpha f_2}(w_t), \\\\
                y_t & = & \\mathrm{prox}_{\\alpha f_1}(2x_t - w_t), \\\\
                w_{t+1} & = & w_t + \\theta (y_t - x_t).
            \\end{eqnarray}

    **Theoretical guarantee**:

    The **tight** theoretial guarantee is obtained in [2, Theorem 2]:

        .. math:: \|w_1 - w_1'\|^2 \\leqslant  \\max\\left(\\frac{1}{1 + \\mu \\alpha}, \\frac{\\alpha L }{1 + L \\alpha}\\right)^{2n} \|w_0 - w_0'\|^2

    for when :math:`\\theta=1`.

    **References**:

    Details on the SDP formulations can be found in

    `[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020). Operator splitting performance estimation:
    Tight contraction factors and optimal parameter selection. SIAM Journal on Optimization, 30(3), 2251-2271.
    <https://arxiv.org/pdf/1812.00146.pdf>`_

    When :math:`\\theta = 1`, the bound can be compared with that of [2, Theorem 2]

    `[2] P. Giselsson, and S. Boyd (2016). Linear convergence and metric selection in
    Douglas-Rachford splitting and ADMM. IEEE Transactions on Automatic Control, 62(2), 532-544.
    <https://arxiv.org/pdf/1410.8479.pdf>`_

    A description for the DRS method can be found in [3, 7.3]

    `[3] E. Ryu, S. Boyd (2016). A primer on monotone operator methods.
    Applied and Computational Mathematics 15(1), 3-43.
    <https://web.stanford.edu/~boyd/papers/pdf/monotone_primer.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        alpha (float): parameter of the scheme.
        theta (float): parameter of the scheme.
        n (int): number of iterations.
        verbose (bool, optional): if True, print conclusion.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Examples:
        >>> pepit_tau, theoretical_tau = wc_douglas_rachford_splitting_contraction(mu=.1, L=1, alpha=3, theta=1, n=2, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 12x12
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
                 function 1 : 20 constraint(s) added
                 function 2 : 20 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.35012779919911946
        *** Example file: worst-case performance of the Douglas Rachford Splitting in distance ***
            PEPit guarantee:		 ||w - wp||^2 <= 0.350128 ||w0 - w0p||^2
            Theoretical guarantee:	 ||w - wp||^2 <= 0.350128 ||w0 - w0p||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex and a smooth strongly convex function.
    func1 = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})
    func2 = problem.declare_function(ConvexFunction, param={})

    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting points w0 and w0p of the algorithm
    w0 = problem.set_initial_point()
    w0p = problem.set_initial_point()

    # Set the initial constraint that is the distance between w0 and w0p
    problem.set_initial_condition((w0 - w0p) ** 2 <= 1)

    # Compute n steps of the Douglas Rachford Splitting starting from w0
    w = w0
    for _ in range(n):
        x, _, _ = proximal_step(w, func2, alpha)
        y, _, _ = proximal_step(2 * x - w, func1, alpha)
        w = w + theta * (y - x)

    # Compute n steps of the Douglas Rachford Splitting starting from w0p
    wp = w0p
    for _ in range(n):
        xp, _, _ = proximal_step(wp, func2, alpha)
        yp, _, _ = proximal_step(2 * xp - wp, func1, alpha)
        wp = wp + theta * (yp - xp)

    # Set the performance metric to the final distance between w and wp
    problem.set_performance_metric((w - wp) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison) when theta = 1
    if theta == 1:
        theoretical_tau = (max(1 / (1 + mu * alpha), alpha * L / (1 + alpha * L))) ** (2 * n)
    else:
        theoretical_tau = None

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Douglas-Rachford splitting in distance ***')
        print('\tPEPit guarantee:\t ||w - wp||^2 <= {:.6} ||w0 - w0p||^2'.format(pepit_tau))
        if theta == 1:
            print('\tTheoretical guarantee:\t ||w - wp||^2 <= {:.6} ||w0 - w0p||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_douglas_rachford_splitting_contraction(mu=.1, L=1, alpha=3, theta=1, n=2, verbose=True)
