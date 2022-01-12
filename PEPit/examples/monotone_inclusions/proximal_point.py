from PEPit import PEP
from PEPit.operators import MonotoneOperator
from PEPit.primitive_steps import proximal_step


def wc_proximal_point(alpha, n, verbose=True):
    """
    Consider the monotone inclusion problem

        .. math:: \\mathrm{Find}\\, x:\\, 0\\in Ax,

    where :math:`A` is maximally monotone. We denote :math:`J_A = (I + A)^{-1}` the resolvents of :math:`A`.

    This code computes a worst-case guarantee for the **proximal point** method.
    That, it computes the smallest possible :math:`\\tau(n, \\alpha)` such that the guarantee

        .. math:: \\|x_n - x_{n-1}\\|^2 \\leqslant \\tau(n, \\alpha) \\|x_0 - x_\\star\\|^2,

    is valid, where :math:`x_\\star` is such that :math:`0 \\in Ax_\\star`.

    **Algorithm**: The proximal point algorithm for monotone inclusions is described as follows, for :math:`t \in \\{ 0, \\dots, n-1\\}`,

        .. math:: x_{t+1} = J_{\\alpha A}(x_t),

    where :math:`\\alpha` is a step-size.

    **Theoretical guarantee**: A tight theoretical guarantee can be found in [1, section 4].

        .. math:: \\|x_n - x_{n-1}\\|^2 \\leqslant \\frac{\\left(1 - \\frac{1}{n}\\right)^{n - 1}}{n} \\|x_0 - x_\\star\\|^2.

    **Reference**:

    `[1] G. Gu, J. Yang (2020). Tight sublinear convergence rate of the proximal point algorithm for maximal
    monotone inclusion problem. SIAM Journal on Optimization, 30(3), 1905-1921.
    <https://epubs.siam.org/doi/pdf/10.1137/19M1299049>`_

    Args:
        alpha (float): the step-size.
        n (int): number of iterations.
        verbose (bool, optional): if True, print conclusion.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (float): theoretical value.

    Example:
        >>> pepit_tau, theoretical_tau = wc_proximal_point(alpha=2, n=10, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 12x12
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 110 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.03874199421010509
        *** Example file: worst-case performance of the Proximal Point Method***
            PEPit guarantee:		 ||x(n) - x(n-1)||^2 <= 0.038742 ||x0 - xs||^2
            Theoretical guarantee:	 ||x(n) - x(n-1)||^2 <= 0.038742 ||x0 - xs||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a monotone operator
    A = problem.declare_function(MonotoneOperator, param={})

    # Start by defining its unique optimal point xs = x_*
    xs = A.stationary_point()

    # Then define the starting point x0 of the algorithm and its gradient value g0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute n steps of the Proximal Gradient method starting from x0
    x = x0
    for _ in range(n):
        previous_x = x
        x, _, _ = proximal_step(previous_x, A, alpha)

    # Set the performance metric to the distance between x(n) and x(n-1)
    problem.set_performance_metric((x - previous_x) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (1 - 1 / n) ** (n - 1) / n

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Proximal Point Method***')
        print('\tPEPit guarantee:\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_proximal_point(alpha=2, n=10, verbose=True)
