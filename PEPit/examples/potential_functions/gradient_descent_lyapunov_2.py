from PEPit.pep import PEP
from PEPit.functions.smooth_convex_function import SmoothConvexFunction


def wc_gradient_descent_lyapunov_2(L, gamma, n, verbose=True):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code computes a worst-case guarantee for **gradient descent** with fixed step-size :math:`\\gamma`,
    for a well-chosen Lyapunov function:

    .. math:: V_t = (2t + 1) L \\left(f(x_t) - f_\\star\\right) + t(t+2) \\|\\nabla f(x_t)\\|^2 + L^2 \\|x_t - x_\\star\\|^2

    That is, it verifies that the above Lyapunov is decreasing on the trajectory:

    .. math :: V_{t+1} \\leq V_t

    is valid, where :math:`x_t` is the :math:`t^{\\mathrm{th}}`
    output of the **gradient descent** with fixed step-size :math:`\\frac{1}{L}`.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma \\nabla f(x_t),

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**:
    The theoretical guarantee can be found in [1, Theorem 3]:

    .. math:: V_{t+1} \\leq V_t,

    when :math:`\\gamma=\\frac{1}{L}`.

    References:

        The detailed potential approach and the SDP approach are available in:

        `[1] Adrien Taylor, and Francis Bach. "Stochastic first-order
        methods: non-asymptotic and computer-aided analyses via potential functions." (2019)
        <https://arxiv.org/pdf/1902.00947.pdf>`_

    Args:
        L (float): the smoothness parameter.
        gamma (float): the step-size.
        n (int): rank of studied iteration.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Examples:
        >>> L = 1
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_lyapunov_2(L=L, gamma=1 / L, n=10, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 4x4
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (0 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 6 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 1.894425729310791e-17
        *** Example file: worst-case performance of gradient descent with fixed step-size for a given Lyapunov function***
            PEP-it guarantee:		[(2t + 3)L*(f(x_(t+1)) - f_*) + (t+1)(t+3) ||f'(x_(t+1))||^2 + L^2 ||x_(t+1) - x_*||^2] - [(2t + 1)L*(f(x_t) - f_*) + t(t+2) ||f'(x_t)||^2 + L^2 ||x_t - x_*||^2] <= 1.89443e-17
            Theoretical guarantee:	[(2t + 3)L*(f(x_(t+1)) - f_*) + (t+1)(t+3) ||f'(x_(t+1))||^2 + L^2 ||x_(t+1) - x_*||^2] - [(2t + 1)L*(f(x_t) - f_*) + t(t+2) ||f'(x_t)||^2 + L^2 ||x_t - x_*||^2] <= 0.0

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
    init_lyapunov = (2 * n + 1) * L * (fn - fs) + n * (n + 2) * gn ** 2 + L ** 2 * (xn - xs) ** 2
    final_lyapunov = (2 * n + 3) * L * (fnp1 - fs) + (n + 1) * (n + 3) * gnp1 ** 2 + L ** 2 * (xnp1 - xs) ** 2

    # Set the performance metric to the difference between the initial and the final Lyapunov
    problem.set_performance_metric(final_lyapunov - init_lyapunov)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 0.

    # Print conclusion if required
    if verbose:
        print('*** Example file:'
              ' worst-case performance of gradient descent with fixed step-size for a given Lyapunov function***')
        print('\tPEP-it guarantee:\t\t'
              '[(2t + 3)L*(f(x_(t+1)) - f_*) + (t+1)(t+3) ||f\'(x_(t+1))||^2 + L^2 ||x_(t+1) - x_*||^2]'
              ' - '
              '[(2t + 1)L*(f(x_t) - f_*) + t(t+2) ||f\'(x_t)||^2 + L^2 ||x_t - x_*||^2]'
              ' <= {:.6}'.format(pepit_tau))
        print('\tTheoretical guarantee:\t'
              '[(2t + 3)L*(f(x_(t+1)) - f_*) + (t+1)(t+3) ||f\'(x_(t+1))||^2 + L^2 ||x_(t+1) - x_*||^2]'
              ' - '
              '[(2t + 1)L*(f(x_t) - f_*) + t(t+2) ||f\'(x_t)||^2 + L^2 ||x_t - x_*||^2]'
              ' <= {:.6}'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    L = 1
    pepit_tau, theoretical_tau = wc_gradient_descent_lyapunov_2(L=L, gamma=1 / L, n=10, verbose=True)
