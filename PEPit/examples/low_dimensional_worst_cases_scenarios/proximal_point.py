from PEPit import PEP
from PEPit.operators import MonotoneOperator
from PEPit.primitive_steps import proximal_step


def wc_proximal_point(alpha, n, verbose=1):
    """
    Consider the monotone inclusion problem

        .. math:: \\mathrm{Find}\\, x:\\, 0\\in Ax,

    where :math:`A` is maximally monotone. We denote :math:`J_A = (I + A)^{-1}` the resolvents of :math:`A`.

    This code computes a worst-case guarantee for the **proximal point** method.
    That, it computes the smallest possible :math:`\\tau(n, \\alpha)` such that the guarantee

        .. math:: \\|x_n - x_{n-1}\\|^2 \\leqslant \\tau(n, \\alpha) \\|x_0 - x_\\star\\|^2,

    is valid, where :math:`x_\\star` is such that :math:`0 \\in Ax_\\star`.
    This example further illustrates how to find a low-dimensional worst-case example, recovering that of [1].

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
        verbose (int): Level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (float): theoretical value.

    Example:
        >>> pepit_tau, theoretical_tau = wc_proximal_point(alpha=2.2, n=11, verbose=1)
        (PEPit) Setting up the problem: size of the main PSD matrix: 13x13
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
		 function 1 : 132 constraint(s) added
	(PEPit) Setting up the problem: 0 lmi constraint(s) added
	(PEPit) Compiling SDP
	(PEPit) Calling SDP solver
	(PEPit) Solver status: optimal (solver: MOSEK); optimal value: 0.03504938911860289
	(PEPit) Postprocessing: 3 eigenvalue(s) > 6.668973163878725e-09 before dimension reduction
	(PEPit) Calling SDP solver
	(PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.03504938911860289
	(PEPit) Postprocessing: 2 eigenvalue(s) > 4.190375858424204e-09 after dimension reduction
	*** Example file: worst-case performance of the Proximal Point Method***
		PEPit guarantee:	 ||x(n) - x(n-1)||^2 <= 0.0350394 ||x0 - xs||^2
		Theoretical guarantee:	 ||x(n) - x(n-1)||^2 <= 0.0350494 ||x0 - xs||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a monotone operator
    A = problem.declare_function(MonotoneOperator)

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
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose, dimension_reduction_heuristic="trace")

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (1 - 1 / n) ** (n - 1) / n

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the Proximal Point Method***')
        print('\tPEPit guarantee:\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_proximal_point(alpha=2.2, n=11, verbose=1)