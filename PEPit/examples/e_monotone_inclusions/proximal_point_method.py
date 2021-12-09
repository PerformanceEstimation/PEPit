from PEPit.pep import PEP
from PEPit.operators.monotone import MonotoneOperator
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_ppm(alpha, n, verbose=True):
    """
    Consider the monotone inclusion problem

        .. math:: \\text{Find} \ x, 0 \in Ax,

    where :math:`A` is maximally monotone. We denote :math:`J_A = (I + A)^{-1}` the resolvents of :math:`A`.

    This code computes a worst-case guarantee for the **proximal point** method,
    that is the smallest possible :math:`\\tau(n, \\alpha)` such that the guarantee

        .. math:: ||x_n - y_n||^2 \\leqslant \\tau(n, \\alpha) ||x_0 - x_\star||^2,

    is valid, where :math:`x_\star` is such that :math:`0 \\in Ax_\star`.

    **Algorithm**:

        .. math::

            \\begin{eqnarray}
                x_{i+1} & = & J_{\\alpha A}(y_i) \\\\
                y_{i+1} & = & x_{i+1} + \\frac{i}{i+2}(x_{i+1} - x_{i}) - \\frac{i}{i+1}(x_i - y_{i-1})
            \\end{eqnarray}

    **Theoretical guarantee**:

    Theoretical rates can be found in [1, section 4].

        .. math:: \\|x_n - x_{n-1}\\|^2 \\leqslant  \\frac{\\left(1 - \\frac{1}{n}\\right)^{n - 1}}{n} \\|x_0 - x_\star\\|^2

    **Reference**:

        Theoretical rates can be found in [1, section 4].

        [1] Guoyong Gu, and Junfeng Yang. "Optimal nonergodic sublinear
        convergence rate of proximal point algorithm for maximal monotone
        inclusion problems." (2019)

    Args:
        alpha (float): the step size
        n (int): number of iterations.
        verbose (bool, optional): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_ppm(alpha=2, n=10, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 12x12
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 110 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.03874199421010509
        *** Example file: worst-case performance of the Proximal Point Method***
            PEP-it guarantee:		 ||x(n) - x(n-1)||^2 <= 0.038742 ||x0 - xs||^2
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
        print('\tPEP-it guarantee:\t\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_ppm(alpha=2, n=10, verbose=True)
