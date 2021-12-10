from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.functions.convex_function import ConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_pgd(L, mu, gamma, n, verbose=True):
    """
    Consider the composite convex minimization problem

    .. math:: f_\\star = \\min_x {F(x) \\equiv f_1(x) + f_2(x)},

    where :math:`f_1` is :math:`L`-smooth and :math:`\\mu`-strongly convex,
    and where :math:`f_2` is closed convex and proper.

    This code computes a worst-case guarantee for the **proximal gradient** method.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu)` such that the guarantee

    .. math :: \\|x_n - x_\\star\\|^2 \\leqslant \\tau(n, L, \\mu) \\|x_0 - x_\\star\\|^2,

    is valid, where :math:`x_n` is the output of the **proximal gradient**,
    and where :math:`x_\\star` is a minimizer of :math:`F`.
    In short, for given values of :math:`n`, :math:`L` and :math:`\\mu`,
    :math:`\\tau(n, L, \\mu)` is computed as the worst-case value of
    :math:`\\|x_n - x_\\star\\|^2` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:

        .. math::
            \\begin{eqnarray}
                y_t & = & x_t - \\gamma \\nabla f_1(x_t) \\\\
                x_{t+1} & = & \\arg\\min_x \\left\\{f_2(x)+\\frac{1}{2\gamma}||x-y_t||^2 \\right\\},
            \\end{eqnarray}

        where :math:`\\gamma` is a step size.

    **Theoretical guarantee**:

        TODO
        The **?** guarantee obtained in ?? is

        .. math:: \\tau(n, L, \\mu) =

    References:
        TODO: Check reference, I find proximal point, fast proximal gradient and plenty of others but not proximal gradient here:
        `[1] A. Taylor, J. Hendrickx, F. Glineur (2017). Exact worst-case performance of first-order methods for
        composite convex optimization. SIAM Journal on Optimization, 27(3):1283–1313.
        <https://arxiv.org/pdf/1512.07516.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        gamma (float): proximal step size
        n (int): number of iterations.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_pgd(L=1, mu=.1, gamma=1, n=2, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 7x7
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 2 function(s)
                 function 1 : 6 constraint(s) added
                 function 2 : 6 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.6560999999942829
        *** Example file: worst-case performance of the Proximal Gradient Method in function values***
            PEP-it guarantee:	 ||x_n - x_*||^2 <= 0.6561 ||x0 - xs||^2
            Theoretical guarantee :	 ||x_n - x_*||^2 <= 0.6561 ||x0 - xs||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function and a closed convex proper function
    f1 = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})
    f2 = problem.declare_function(ConvexFunction, param={})
    func = f1 + f2

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the proximal gradient method starting from x0
    x = x0
    for _ in range(n):
        y = x - gamma * f1.gradient(x)
        x, _, _ = proximal_step(y, f2, gamma)

    # Set the performance metric to the distance between x and xs
    problem.set_performance_metric((x - xs) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = max((1 - mu*gamma)**2, (1 - L*gamma)**2)**n

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Proximal Gradient Method in function values***')
        print('\tPEP-it guarantee:\t ||x_n - x_*||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee :\t ||x_n - x_*||^2 <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_pgd(L=1, mu=.1, gamma=1, n=2, verbose=True)
