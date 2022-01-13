from math import sqrt

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def wc_triple_momentum(mu, L, n, verbose=True):
    """
    Consider the minimization problem

        .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for **triple momentum method** (TMM).
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu)` such that the guarantee

        .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L, \\mu) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the TMM, and where :math:`x_\\star` is the minimizer of :math:`f`.
    In short, for given values of :math:`n`, :math:`L` and :math:`\\mu`, :math:`\\tau(n, L, \\mu)` is computed
    as the worst-case value of :math:`f(x_n)-f_\\star` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.


    **Algorithm**:

    For :math:`t \in \\{ 1, \dots, n\\}`

        .. math::
            :nowrap:

            \\begin{eqnarray}
               \\xi_{t+1} &&= (1 + \\beta)  \\xi_{t} - \\beta  \\xi_{t-1} - \\alpha \\nabla f(y_t) \\\\
               y_{t} &&= (1+\\gamma ) \\xi_{t} -\\gamma \\xi_{t-1} \\\\
               x_{t} && = (1 + \\delta)  \\xi_{t} - \\delta \\xi_{t-1}
            \\end{eqnarray}

    with

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\kappa &&= \\frac{L}{\\mu} , \\quad \\rho = 1- \\frac{1}{\\sqrt{\\kappa}}\\\\
                (\\alpha, \\beta, \\gamma,\\delta) && = \\left(\\frac{1+\\rho}{L}, \\frac{\\rho^2}{2-\\rho},
                \\frac{\\rho^2}{(1+\\rho)(2-\\rho)}, \\frac{\\rho^2}{1-\\rho^2}\\right)
            \\end{eqnarray}

    and

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\xi_{0} = x_0 \\\\
                \\xi_{1} = x_0 \\\\
                y = x_0
            \\end{eqnarray}


    **Theoretical guarantee**:
    A theoretical **upper** (empirically tight) bound can be found in [1, Theorem 1, eq. 4]:

        .. math:: f(x_n)-f_\\star \\leqslant \\frac{\\rho^{2(n+1)} L \\kappa}{2}\\|x_0 - x_\\star\\|^2.

    **References**:
    The triple momentum method was discovered and analyzed in [1].

    `[1] Van Scoy, B., Freeman, R. A., Lynch, K. M. (2018), The fastest known globally convergent first-order method for
    minimizing strongly convex functions. IEEE Control Systems Letters, 2(1), 49-54.
    <http://www.optimization-online.org/DB_FILE/2017/03/5908.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value


    Example:
        >>> pepit_tau, theoretical_tau = wc_triple_momentum(mu=0.1, L=1., n=4, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 7x7
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 30 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.23893532450841679
        *** Example file: worst-case performance of the Triple Momentum Method ***
            PEPit guarantee:		 f(x_n)-f_* <= 0.238935 ||x_0-x_*||^2
            Theoretical guarantee:	 f(x_n)-f_* <= 0.238925 ||x_0-x_*||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex
    func = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Set the parameters of the Triple Momentum Method
    kappa = L / mu
    rho = (1 - 1 / sqrt(kappa))
    alpha = (1 + rho) / L
    beta = rho ** 2 / (2 - rho)
    gamma = rho ** 2 / (1 + rho) / (2 - rho)
    delta = rho ** 2 / (1 - rho ** 2)

    # Run n steps of the Triple Momentum Method
    x_old = x0
    x_new = x0
    y = x0
    for _ in range(n):
        x_inter = (1 + beta) * x_new - beta * x_old - alpha * func.gradient(y)
        y = (1 + gamma) * x_inter - gamma * x_new
        x = (1 + delta) * x_inter - delta * x_new
        x_new, x_old = x_inter, x_new

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func.value(x) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = rho ** (2 * n) * L / 2 * kappa

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Triple Momentum Method ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0-x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0-x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_triple_momentum(mu=0.1, L=1., n=4, verbose=True)
