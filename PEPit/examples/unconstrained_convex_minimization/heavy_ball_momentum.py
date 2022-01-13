from math import sqrt

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def wc_heavy_ball_momentum(mu, L, alpha, beta, n, verbose=True):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for the **Heavy-ball (HB)** method, aka **Polyak momentum** method.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L, \\mu) (f(x_0) - f_\\star)

    is valid, where :math:`x_n` is the output of the **Heavy-ball (HB)** method,
    and where :math:`x_\\star` is the minimizer of :math:`f`.
    In short, for given values of :math:`n`, :math:`L` and :math:`\\mu`,
    :math:`\\tau(n, L, \\mu)` is computed as the worst-case value of
    :math:`f(x_n)-f_\\star` when :math:`f(x_0) - f_\\star \\leqslant 1`.

    **Algorithm**:

        .. math:: x_{t+1} = x_t - \\alpha \\nabla f(x_t) + \\beta (x_t-x_{t-1})

        with

        .. math:: \\alpha \\in (0, \\frac{1}{L}]

        and

        .. math:: \\beta = \\sqrt{(1 - \\alpha \\mu)(1 - L \\alpha)}

    **Theoretical guarantee**:

    The **upper** guarantee obtained in [2, Theorem 4] is

        .. math:: f(x_n) - f_\\star \\leqslant (1 - \\alpha \\mu)^n (f(x_0) - f_\\star).

    **References**: This methods was first introduce in [1, Section 2], and convergence upper bound was proven in [2, Theorem 4].

    `[1] B.T. Polyak (1964). Some methods of speeding up the convergence of iteration method.
    URSS Computational Mathematics and Mathematical Physics.
    <https://www.sciencedirect.com/science/article/pii/0041555364901375>`_

    `[2] E. Ghadimi, H. R. Feyzmahdavian, M. Johansson (2015). Global convergence of the Heavy-ball method for
    convex optimization. European Control Conference (ECC).
    <https://arxiv.org/pdf/1412.7457.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        alpha (float): parameter of the scheme.
        beta (float): parameter of the scheme such that :math:`0<\\beta<1` and :math:`0<\\alpha<2(1+\\beta)`.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> mu = 0.1
        >>> L = 1.
        >>> alpha = 1 / (2 * L)  # alpha \in [0, 1 / L]
        >>> beta = sqrt((1 - alpha * mu) * (1 - L * alpha))
        >>> pepit_tau, theoretical_tau = wc_heavy_ball_momentum(mu=mu, L=L, alpha=alpha, beta=beta, n=2, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 5x5
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 12 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.8145062493468549
        *** Example file: worst-case performance of the Heavy-Ball method ***
            PEPit guarantee:		 f(x_n)-f_* <= 0.753492 (f(x_0) -  f(x_*))
            Theoretical guarantee:	 f(x_n)-f_* <= 0.9025 (f(x_0) -  f(x_*))

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding function value f0
    x0 = problem.set_initial_point()
    f0 = func.value(x0)

    # Set the initial constraint that is the distance between f(x0) and f(x^*)
    problem.set_initial_condition((f0 - fs) <= 1)

    # Run one step of the heavy ball method
    x_new = x0
    x_old = x0

    for _ in range(n):
        x_next = x_new - alpha * func.gradient(x_new) + beta * (x_new - x_old)
        x_old = x_new
        x_new = x_next

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric(func.value(x_new) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (1 - alpha * mu) ** n

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Heavy-Ball method ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0) -  f(x_*))'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0) -  f(x_*))'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    mu = 0.1
    L = 1.
    alpha = 1 / (2 * L)  # alpha \in [0, 1 / L]
    beta = sqrt((1 - alpha * mu) * (1 - L * alpha))
    pepit_tau, theoretical_tau = wc_heavy_ball_momentum(mu=mu, L=L, alpha=alpha, beta=beta, n=2, verbose=True)
