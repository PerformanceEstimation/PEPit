from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.functions.convex_function import ConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_drs(mu, L, alpha, theta, n, verbose=True):
    """
    Consider the composite convex minimization problem

        .. math:: \min_x \\{F(x) \equiv f_1(x)+f_2(x) \\}

    where :math:`f_1(x)` is :math:`L`-smooth and :math:`\mu`-strongly convex, and :math:`f_2` is convex,
    closed and proper. Both proximal operators are assumed to be available.

    This code computes a worst-case guarantee for the **Douglas Rachford Splitting (DRS)** method.
    That is, it computes the smallest possible :math:`\\tau(\\mu,L,\\alpha,\\theta,n)` such that the guarantee

        .. math:: \|w_1 - w_1'\|^2 \\leqslant \\tau(\\mu,L,\\alpha,\\theta,n) \|w_0 - w_0'\|^2.

    is valid, where :math:`x_n` is the output of the **Douglas Rachford Splitting method**. It is a contraction
    factor computed when the algorithm is started from two different points :math:`w_0` and :math:`w_0`.

    **Algorithm**:

    Our notations for the DRS algorithm are as follows

        .. math:: x_k     = \\mathrm{prox}_{\\alpha f_2}(w_k)
        .. math:: y_k     = \\mathrm{prox}_{\\alpha f_1}(2x_k - w_k)
        .. math:: w_{k+1} = w_k +\\theta (y_k - x_k)

    **Theoretical guarantee**:

    The **tight** theoretial guarantee is obtained in [2, Theorem 2]:

        .. math:: \|w_1 - w_1'\|^2 \\leqslant  \\max\\left(\\frac{1}{1 + \\mu \\alpha}, \\frac{\\alpha L }{1 + L \\alpha}\\right)^{2n} \|w_0 - w_0'\|^2

    **References**:

    Details on the SDP formulations can be found in

    [1] Ernest K. Ryu, Adrien B. Taylor, Carolina Bergeling, and Pontus Giselsson. "Operator splitting
    performance estimation: Tight contraction factors and optimal parameter selection." (2018)

    When :math:`\\theta = 1`, the bound can be compared with that of [2, Theorem 2]

    [2] Giselsson, Pontus, and Stephen Boyd. "Linear convergence and metric selection in
    Douglas-Rachford splitting and ADMM." IEEE Transactions on Automatic Control (2016).

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        alpha (float): parameter of the scheme.
        theta (float): parameter of the scheme.
        n (int): number of iterations.
        verbose (bool, optional): if True, print conclusion.

    Returns:
        tuple: worst_case value, theoretical value

    Examples:
        >>> pepit_tau, theoretical_tau = wc_drs(0.1, 1, 3, 1, 2)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 12x12
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 2 function(s)
                 function 1 : 20 constraint(s) added
                 function 2 : 20 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.35012898383515983
        *** Example file: worst-case performance of the Douglas Rachford Splitting in distance ***
            PEP-it guarantee:	 ||w - wp||^2 <= 0.350129 ||w0 - w0p||^2
            Theoretical guarantee:	||w - wp||^2 <= 0.350128 ||w0 - w0p||^2
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

    # Then define the starting points x0 and x0p of the algorithm
    w0 = problem.set_initial_point()
    w0p = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x0p
    problem.set_initial_condition((w0 - w0p) ** 2 <= 1)

    # Compute n steps of the Douglas Rachford Splitting starting from x0
    w = w0
    for _ in range(n):
        x, _, _ = proximal_step(w, func2, alpha)
        y, _, _ = proximal_step(2 * x - w, func1, alpha)
        w = w + theta * (y - x)

    # Compute n steps of the Douglas Rachford Splitting starting from x0p
    wp = w0p
    for _ in range(n):
        xp, _, _ = proximal_step(wp, func2, alpha)
        yp, _, _ = proximal_step(2 * xp - wp, func1, alpha)
        wp = wp + theta * (yp - xp)

    # Set the performance metric to the final distance between wp and w
    problem.set_performance_metric((w - wp) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    # when theta = 1
    theoretical_tau = (max(1 / (1 + mu * alpha), alpha * L / (1 + alpha * L))) ** (2 * n)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Douglas Rachford Splitting in distance ***')
        print('\tPEP-it guarantee:\t ||w - wp||^2 <= {:.6} ||w0 - w0p||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t||w - wp||^2 <= {:.6} ||w0 - w0p||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    mu = 0.1
    L = 1.
    # Test scheme parameters
    alpha = 3
    theta = 1
    n = 2

    pepit_tau, theoretical_tau = wc_drs(mu=mu,
                                        L=L,
                                        alpha=alpha,
                                        theta=theta,
                                        n=n)
