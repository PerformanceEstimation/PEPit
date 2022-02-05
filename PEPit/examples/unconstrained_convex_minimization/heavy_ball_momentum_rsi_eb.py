from PEPit import PEP
from PEPit.operators import RsiEbOperator


def wc_heavy_ball_momentum(mu, L, alpha, beta, n, verbose=True):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`\\mu`-RSI. :math:`L`-EB.

    This code computes a worst-case guarantee for the **Heavy-ball (HB)** method, aka **Polyak momentum** method.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu, \\alpha, \\beta)` such that the guarantee

    .. math:: \\|x_n - x_\\star\||^2 \\leqslant \\tau(n, L, \\mu, \\alpha, \\beta) \\|x_0 - x_\\star\||^2

    is valid, where :math:`x_n` is the output of the **Heavy-ball (HB)** method,
    and where :math:`x_\\star` is the minimizer of :math:`f`.
    In short, for given values of :math:`n`, :math:`L`, :math:`\\mu`, :math:`\\alpha` and :math:`\\beta`,
    :math:`\\tau(n, L, \\mu, \\alpha, \\beta)` is computed as the worst-case value of
    :math:`\\|x_n - x_\\star\||^2` when :math:`\\|x_0 - x_\\star\||^2 \\leqslant 1`.

    **Algorithm**:

        .. math:: x_{t+1} = x_t - \\alpha \\nabla f(x_t) + \\beta (x_t-x_{t-1})

    **Theoretical guarantee**:

    The **upper** guarantee obtained in [TODO] is

        #TODO

    **References**: This methods was first introduce in [1, Section 2], and convergence upper bound was proven in [?].

    `[1] B.T. Polyak (1964). Some methods of speeding up the convergence of iteration method.
    URSS Computational Mathematics and Mathematical Physics.
    <https://www.sciencedirect.com/science/article/pii/0041555364901375>`_

    `[2] TODO
    <TODO>`_

    Args:
        L (float): the EB parameter.
        mu (float): the RSI parameter.
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
        >>> alpha = mu/L**2
        >>> beta = 0
        >>> pepit_tau, theoretical_tau = wc_heavy_ball_momentum(mu=mu, L=L, alpha=alpha, beta=beta, n=5, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 7x7
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 10 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.9509897678489653
        *** Example file: worst-case performance of the Heavy-Ball method ***
            PEPit guarantee:	     f(x_n)-f_* <= 0.95099 (f(x_0) -  f(x_*))
            Theoretical guarantee:	 f(x_n)-f_* <= 1.0 (f(x_0) -  f(x_*))

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(RsiEbOperator, param={'mu': mu, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm as well as corresponding function value f0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between f(x0) and f(x^*)
    problem.set_initial_condition((x0 - xs)**2 <= 1)

    # Run one step of the heavy ball method
    x_new = x0
    x_old = x0

    for _ in range(n):
        x_next = x_new - alpha * func.gradient(x_new) + beta * (x_new - x_old)
        x_old = x_new
        x_new = x_next

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric((x_new - xs)**2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 1.

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Heavy-Ball method ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0) -  f(x_*))'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0) -  f(x_*))'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    mu = 0.1
    L = 1.
    nb = 50

    beta_list = list()
    alpha_list = list()
    pepit_taus = list()
    for beta in np.linspace(0, 1, nb):
        for alpha in np.linspace(0, 4 * mu / L**2, nb):
            beta_list.append(beta)
            alpha_list.append(alpha)
            pepit_tau, _ = wc_heavy_ball_momentum(mu=mu, L=L, alpha=alpha, beta=beta, n=5, verbose=False)
            pepit_taus.append(min(pepit_tau, 1))
    plt.scatter(alpha_list, beta_list, c=pepit_taus)
    plt.colorbar()
    plt.xlabel("alpha")
    plt.ylabel("beta")
    plt.title("Performance of Heavy-Ball algorithm on RSI(.1) EB(1) functions.")
    plt.show()
