from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.primitive_steps import exact_linesearch_step


def wc_gradient_exact_line_search(L, mu, n, verbose=True):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for the **gradient descent** (GD) with **exact linesearch** (ELS).
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L, \\mu) (f(x_0) - f_\\star)

    is valid, where :math:`x_n` is the output of the GD with ELS,
    and where :math:`x_\\star` is the minimizer of :math:`f`.
    In short, for given values of :math:`n`, :math:`L` and :math:`\\mu`,
    :math:`\\tau(n, L, \\mu)` is computed as the worst-case value of
    :math:`f(x_n)-f_\\star` when :math:`f(x_0) - f_\\star \\leqslant 1`.

    **Algorithm**:
    GD with ELS can be written as

        .. math:: x_{t+1} = x_t - \\gamma_t \\nabla f(x_t)

    with :math:`\\gamma_t = \\arg\\min_{\\gamma} f \\left( x_t - \\gamma \\nabla f(x_t) \\right)`.

    **Theoretical guarantee**: The **tight** worst-case guarantee for GD with ELS, obtained in [1, Theorem 1.2], is

        .. math:: f(x_n) - f_\\star \\leqslant \\left(\\frac{L-\\mu}{L+\\mu}\\right)^{2n} (f(x_0) - f_\\star).

    **References**: The detailed approach (based on convex relaxations) is available in [1], along with theoretical bound.

    `[1] E. De Klerk, F. Glineur, A. Taylor (2017). On the worst-case complexity of the gradient method with exact
    line search for smooth strongly convex functions. Optimization Letters, 11(7), 1185-1199.
    <https://link.springer.com/content/pdf/10.1007/s11590-016-1087-4.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        n (int): number of iterations.
        verbose (bool, optional): if True, print conclusion.
    
    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_gradient_exact_line_search(L=1, mu=.1, n=2, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 7x7
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 16 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.44812204883466417
        *** Example file: worst-case performance of gradient descent with exact linesearch (ELS) ***
            PEPit guarantee:		 f(x_n)-f_* <= 0.448122 (f(x_0)-f_*)
            Theoretical guarantee:	 f(x_n)-f_* <= 0.448125 (f(x_0)-f_*)

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value g0 and f0
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Set the initial constraint that is the difference between f0 and f_*
    problem.set_initial_condition(f0 - fs <= 1)

    # Run n steps of GD method with ELS
    x = x0
    gx = g0
    for i in range(n):
        x, gx, fx = exact_linesearch_step(x, func, [gx])

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = ((L - mu) / (L + mu)) ** (2 * n)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of gradient descent with exact linesearch (ELS) ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_gradient_exact_line_search(L=1, mu=.1, n=2, verbose=True)
