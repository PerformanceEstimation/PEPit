from PEPit import PEP
from PEPit.operators import MonotoneOperator
from PEPit.primitive_steps import proximal_step


def wc_accelerated_proximal_point(alpha, n, verbose=True):
    """
    Consider the monotone inclusion problem

        .. math:: \\mathrm{Find}\\, x:\\, 0\\in Ax,

    where :math:`A` is maximally monotone. We denote :math:`J_A = (I + A)^{-1}` the resolvents of :math:`A`.

    This code computes a worst-case guarantee for the **accelerated proximal point** method proposed in [1].
    That, it computes the smallest possible :math:`\\tau(n, \\alpha)` such that the guarantee

        .. math:: \\|x_n - y_n\\|^2 \\leqslant \\tau(n, \\alpha) \\|x_0 - x_\\star\\|^2,

    is valid, where :math:`x_\\star` is such that :math:`0 \\in Ax_\\star`.

    **Algorithm**: Accelerated proximal point is described as follows, for :math:`t \in \\{ 0, \\dots, n-1\\}`

        .. math::

            \\begin{eqnarray}
                x_{t+1} & = & J_{\\alpha A}(y_t), \\\\
                y_{t+1} & = & x_{t+1} + \\frac{t}{t+2}(x_{t+1} - x_{t}) - \\frac{t}{t+1}(x_t - y_{t-1}),
            \\end{eqnarray}

    where :math:`x_0=y_0=y_{-1}`

    **Theoretical guarantee**: A tight theoretical worst-case guarantee can be found in [1, Theorem 4.1], for :math:`n \\geqslant 1`,

        .. math:: \\|x_n - y_{n-1}\\|^2 \\leqslant  \\frac{1}{n^2}  \\|x_0 - x_\\star\\|^2.

    **Reference**:

    `[1] D. Kim (2021). Accelerated proximal point method for maximally monotone operators.
    Mathematical Programming, 1-31.
    <https://arxiv.org/pdf/1905.05149v4.pdf>`_

    Args:
        alpha (float): the step-size
        n (int): number of iterations.
        verbose (bool, optional): if True, print conclusion

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_accelerated_proximal_point(alpha=2, n=10, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 12x12
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 110 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.010000353550061647
        *** Example file: worst-case performance of the Accelerated Proximal Point Method***
            PEPit guarantee:		 ||x_n - y_n||^2 <= 0.0100004 ||x_0 - x_s||^2
            Theoretical guarantee:	 ||x_n - y_n||^2 <= 0.01 ||x_0 - x_s||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a monotone operator
    A = problem.declare_function(MonotoneOperator, param={})

    # Start by defining its unique optimal point xs = x_*
    xs = A.stationary_point()

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute n steps of the Proximal Gradient method starting from x0
    x = [x0 for _ in range(n + 1)]
    y = [x0 for _ in range(n + 1)]
    for i in range(0, n - 1):
        x[i + 1], _, _ = proximal_step(y[i + 1], A, alpha)
        y[i + 2] = x[i + 1] + i / (i + 2) * (x[i + 1] - x[i]) - i / (i + 2) * (x[i] - y[i])
    x[n], _, _ = proximal_step(y[n], A, alpha)

    # Set the performance metric to the distance between xn and yn
    problem.set_performance_metric((x[n] - y[n]) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 1 / n ** 2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Accelerated Proximal Point Method***')
        print('\tPEPit guarantee:\t ||x_n - y_n||^2 <= {:.6} ||x_0 - x_s||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||x_n - y_n||^2 <= {:.6} ||x_0 - x_s||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_accelerated_proximal_point(alpha=2, n=10, verbose=True)
