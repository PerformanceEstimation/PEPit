from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.primitive_steps.exact_linesearch_step import exact_linesearch_step
from PEPit.primitive_steps.inexact_gradient import inexact_gradient


def wc_InexactGrad_ELS(L, mu, epsilon, n, verbose=True):
    """
    Consider the convex minimization problem

        .. math:: f_\star = \min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for an **inexact gradient method with exact linesearch (ELS)**.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu, \\epsilon)` such that the guarantee

        .. math:: f(x_n) - f_\star \\leqslant \\tau(n, L, \\mu, \\epsilon) ( f(x_0) - f_\star )

    is valid, where :math:`x_n` is the output of the **gradient descent with an inexact descent direction and an exact linesearch**,
    and where :math:`x_\star` is the minimizer of :math:`f`.

    The inexact descent direction :math:`d` is assumed to satisfy a relative inaccuracy described by (with :math:`0 \\leqslant \\epsilon < 1`)

        .. math:: || f'(x_i) - d || \\leqslant \\epsilon || f'(x_i) ||,

    where :math:`f'(x_i)` is the true gradient, and d is the approximate descent direction that is used.

    **Algorithm**:

    Select :math:`d_i` such that

        .. math:: \\gamma = \\arg\\min_{\\gamma \in R^d} f(x_i - \\gamma d_i).
        .. math:: x_{i+1} = x_i - \\gamma d_i.

    **Theoretical guarantees**:

    The **tight** guarantee obtained in [1, Theorem 5.1] is

        .. math:: \\tau(n,L,\\mu,\\epsilon) = \\frac{L_{\\epsilon} - \\mu_{\\epsilon}}{L_{\\epsilon} + \\mu_{\\epsilon}}^{2n},

    with :math:`L_{\\epsilon} = (1 + \\epsilon) L` and :math:`\\mu_{\\epsilon} = (1 - \\epsilon) \\mu`

    References:

        The detailed approach (based on convex relaxations) is available in

        [1] De Klerk, Etienne, François Glineur, and Adrien B. Taylor.
        "On the worst-case complexity of the gradient method with exact line search for smooth strongly convex functions."
        Optimization Letters (2017).

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        epsilon (float): level of inaccuracy.
        n (int): number of iterations.
        verbose (bool, optional): if True, print conclusion.

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_InexactGrad_ELS(1, 0.1, 0.1, 1)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 9x9
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 18 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.5186658287501191
        *** Example file: worst-case performance of inexact gradient descent with exact linesearch ***
            PEP-it guarantee:		 f(x_n)-f_* <= 0.518666 (f(x_0)-f_*)
            Theoretical guarantee:	 f(x_n)-f_* <= 0.518917 (f(x_0)-f_*)
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value g0 and f0
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Set the initial constraint that is the distance between f0 and f_*
    problem.set_initial_condition(f0 - fs <= 1)

    # Run n steps of the inexact gradient method with ELS
    x = x0
    for i in range(n):
        dx, _ = inexact_gradient(x, func, epsilon, notion='relative')
        x, gx, fx = exact_linesearch_step(x, func, [dx])

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    Leps = (1 + epsilon) * L
    meps = (1 - epsilon) * mu
    theoretical_tau = ((Leps - meps) / (Leps + meps)) ** (2 * n)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of inexact gradient descent with exact linesearch ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 2
    L = 1
    mu = .1
    epsilon = .1

    pepit_tau, theoretical_tau = wc_InexactGrad_ELS(L=L,
                                                    mu=mu,
                                                    epsilon=epsilon,
                                                    n=n)
