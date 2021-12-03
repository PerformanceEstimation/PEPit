from PEPit.pep import PEP
from PEPit.functions.smooth_convex_function import SmoothConvexFunction


def wc_gd_lyapunov_1(L, gamma, n, verbose=True):
    """
    Consider the convex minimization problem

    .. math:: f_\star = \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code computes a worst-case guarantee for **gradient descent** with fixed step size :math:`\\gamma`,
    for a well-chosen Lyapunov function:

    .. math:: V_k = k*(f(x_k) - f_\\star) + \\frac{L}{2} \\|x_k - x_\\star\\|^2

    That is, it verifies that the above Lyapunov is decreasing on the trajectory:

    .. math :: V_{k+1} \\leq V_k

    is valid, where :math:`x_k` is the :math:`k^{\\mathrm{th}}`
    output of the **gradient descent** with fixed step size :math:`\\frac{1}{L}`.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{k+1} = x_k - \\gamma \\nabla f(x_k),

    where :math:`\\gamma` is a step size.

    **Theoretical guarantee**:
    The theoretical guarantee can be found in [1, Theorem 3.3]:

    .. math:: V_{k+1} \\leq V_k,

    when :math:`\\gamma=\\frac{1}{L}`.

    References:

        The detailed potential approach is available in [1, Theorem 3.3], and the SDP approach in [2].

        `[1] Nikhil Bansal, and Anupam Gupta.  "Potential-function proofs for
        first-order methods." (2019)<https://arxiv.org/pdf/1712.04581.pdf>`_

        `[2] Adrien Taylor, and Francis Bach. "Stochastic first-order
        methods: non-asymptotic and computer-aided analyses via
        potential functions." (2019)<https://arxiv.org/pdf/1902.00947.pdf>`_

    Args:
        L (float): the smoothness parameter.
        gamma (float): the step size.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Examples:
        >>> L = 1
        >>> pepit_tau, theoretical_tau = wc_gd_lyapunov_1(L=L, gamma=1 / L, n=10)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 4x4
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (0 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 6 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 3.3902995517363515e-18
        *** Example file: worst-case performance of gradient descent with fixed step size for a given Lyapunov function***
            PEP-it guarantee:		[(n+1) * (f(x_(n+1)) - f_*) + L / 2 ||x_(n+1) - x_*||^2] - [n * (f(x_n) - f_*) + L / 2 ||x_n - x_*||^2] <= 3.3903e-18
            Theoretical guarantee:	[(n+1) * (f(x_(n+1)) - f_*) + L / 2 ||x_(n+1) - x_*||^2] - [n * (f(x_n) - f_*) + L / 2 ||x_n - x_*||^2] <= 0.0

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction, param={'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value gn and fn
    xn = problem.set_initial_point()
    gn, fn = func.oracle(xn)

    # Run the GD at iteration (n+1)
    xnp1 = xn - gamma * gn
    gnp1, fnp1 = func.oracle(xnp1)

    # Compute the Lyapunov function at iteration n and at iteration n+1
    init_lyapunov = n * (fn - fs) + L / 2 * (xn - xs) ** 2
    final_lyapunov = (n + 1) * (fnp1 - fs) + L / 2 * (xnp1 - xs) ** 2

    # Set the performance metric to the difference between the initial and the final Lyapunov
    problem.set_performance_metric(final_lyapunov - init_lyapunov)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 0.

    # Print conclusion if required
    if verbose:
        print('*** Example file:'
              ' worst-case performance of gradient descent with fixed step size for a given Lyapunov function***')
        print('\tPEP-it guarantee:\t\t'
              '[(n+1) * (f(x_(n+1)) - f_*) + L / 2 ||x_(n+1) - x_*||^2]'
              ' - '
              '[n * (f(x_n) - f_*) + L / 2 ||x_n - x_*||^2] '
              '<= {:.6}'.format(pepit_tau))
        print('\tTheoretical guarantee:\t'
              '[(n+1) * (f(x_(n+1)) - f_*) + L / 2 ||x_(n+1) - x_*||^2]'
              ' - '
              '[n * (f(x_n) - f_*) + L / 2 ||x_n - x_*||^2] '
              '<= {:.6}'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    L = 1
    pepit_tau, theoretical_tau = wc_gd_lyapunov_1(L=L, gamma=1 / L, n=10)
