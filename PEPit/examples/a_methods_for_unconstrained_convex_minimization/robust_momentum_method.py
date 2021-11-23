import numpy as np

from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_rmm(mu, L, lam, verbose=True):
    """
    Consider the convex minimization problem

        .. math:: f_* = min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly-convex.

    This code computes a worst-case guarantee for the **robust momentum method**.
    That is, it verifies that the guarantee

        .. math:: v(x_{n+1}) \\leqslant v(x_{n}),

    is valid, where :math:`x_n` is the output of the **robust momentum method**, where :math:`x_*` is a minimizer of :math:`f`,
    and where :math:`v(x_n)` is a well-chosen Lyapunov function decreasing along the sequence

        .. math:: \\kappa = \\frac{\\mu}{L}

        .. math:: \\rho = lam (1 - \\frac{1}{\\kappa}) + (1 - lam) (1 - \\sqrt{\\frac{1}{\\kappa}})

        .. math:: l = \\mu^2  \\frac{\\kappa - \\kappa \\rho^2 - 1}{2 \\rho (1 - \\rho)}

        .. math:: q_n = (L - \\mu) (f(x_n) - f(x_* - \\frac{\\mu}{2}||y_n - x_*||^2 - \\frac{1}{2}||\\nabla(y_n) - \\mu (y_n - x_*)||^2

        .. math:: v(x_n) = l||z_n - x_*||^2 + q_n

    **Algorithm**:

        .. math:: x_{n+1} = x_{n} + \\beta (x_n - x_{n-1}) - \\alpha \\nabla f(y_n)

        .. math:: y_{n} + \\gamma (x_n - x_{n-1})

    with :math:`\\kappa = \\frac{\\mu}{L}`, :math:`\\alpha = \\frac{\\kappa (1 - \\rho^2)(1 + \\rho)}{L}`,

     :math:`\\beta = \\frac{\\kappa \\rho^3}{\\kappa - 1}` and :math:`\\gamma = \\frac{\\rho^2}{(\\kappa - 1)(1 - \\rho)^2(1 + \\rho)}`.
    
    **Theoretical guarantee**:

    The **tight** bound is obtained in [1, Theorem 1],
    
        .. math:: \\tau(n, \\mu, L) = 1

        .. math:: \\rho = lam (1 - \\frac{1}{\\kappa}) + (1 - lam) (1 - \\sqrt{\\frac{1}{\\kappa}})
    
    **References**:

    We show how to compute the tight rate for the Lyapunov function developed in [1, Theorem 1]

    [1] Cyrus, S., Hu, B., Van Scoy, B., & Lessard, L. "A robust accelerated
    optimization algorithm for strongly convex functions." In 2018 Annual
    American Control Conference (ACC) (pp. 1376-1381). IEEE.
         
    Args:    
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        lam (float): if :math:`lam=1` it is the gradient descent, if :math:`lam=0`, it is the Triple Momentum Method.
        verbose (bool, optional): if True, print conclusion

    Returns:
         tuple: worst_case value, theoretical value
    
    Examples:
        >>> pepit_tau, theoretical_tau = wc_rmm(0.1, 1, 0.2)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 5x5
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 6 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.5285548355257013
        *** Example file: worst-case performance of the Robust Momentum Method ***
            PEP-it guarantee:		 v(x_(n+1)) <= 0.528555 v(x_n)
            Theoretical guarantee:	 v(x_(n+1)) <= 0.528555 v(x_n)
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then Define the starting points of the algorithm, x0 and x1
    x0 = problem.set_initial_point()
    x1 = problem.set_initial_point()

    # Set the parameters of the robust momentum method
    kappa = L / mu
    rho = lam * (1 - 1 / kappa) + (1 - lam) * (1 - 1 / float(np.sqrt(kappa)))
    alpha = kappa * (1 - rho) ** 2 * (1 + rho) / L
    beta = kappa * rho ** 3 / (kappa - 1)
    gamma = rho ** 3 / ((kappa - 1) * (1 - rho) ** 2 * (1 + rho))
    l = mu ** 2 * (kappa - kappa * rho ** 2 - 1) / (2 * rho * (1 - rho))
    nnu = (1 + rho) * (1 - kappa + 2 * kappa * rho - kappa * rho ** 2) / (2 * rho)

    # Run one step of the Robust Momentum Method
    y0 = x1 + gamma * (x1 - x0)
    g0, f0 = func.oracle(y0)
    x2 = x1 + beta * (x1 - x0) - alpha * g0
    y1 = x2 + gamma * (x2 - x1)
    g1, f1 = func.oracle(y1)
    x3 = x2 + beta * (x2 - x1) - alpha * g1

    z1 = (x2 - (rho ** 2) * x1) / (1 - rho ** 2)
    z2 = (x3 - (rho ** 2) * x2) / (1 - rho ** 2)

    # Evaluate the lyapunov function at the first and second iterates
    q0 = (L - mu) * (f0 - fs - mu / 2 * (y0 - xs) ** 2) - 1 / 2 * (g0 - mu * (y0 - xs)) ** 2
    q1 = (L - mu) * (f1 - fs - mu / 2 * (y1 - xs) ** 2) - 1 / 2 * (g1 - mu * (y1 - xs)) ** 2
    initLyapunov = l * (z1 - xs) ** 2 + q0
    finalLyapunov = l * (z2 - xs) ** 2 + q1

    # Set the initial constraint that is a bound on the initial Lyapunov function
    problem.set_initial_condition(initLyapunov <= 1)

    # Set the performance metric to the final distance to optimum, that is the final Lyapunov function
    problem.set_performance_metric(finalLyapunov)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = rho ** 2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Robust Momentum Method ***')
        print('\tPEP-it guarantee:\t\t v(x_(n+1)) <= {:.6} v(x_n)'.format(
            pepit_tau))
        print('\tTheoretical guarantee:\t v(x_(n+1)) <= {:.6} v(x_n)'.format(
            theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    mu = 0.1
    L = 1.
    lam = 0.2

    pepit_tau, theoretical_tau = wc_rmm(mu=mu,
                                        L=L,
                                        lam=lam)
