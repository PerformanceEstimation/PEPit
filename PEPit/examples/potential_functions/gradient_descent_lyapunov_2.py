from PEPit import PEP
from PEPit.functions import SmoothConvexFunction


def wc_gradient_descent_lyapunov_2(L, gamma, n, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code verifies a worst-case guarantee for **gradient descent** with fixed step-size :math:`\\gamma`.
    That is, it verifies that the Lyapunov (or potential/energy) function

    .. math:: V_n \\triangleq (2n + 1) L \\left(f(x_n) - f_\\star\\right) + n(n+2) \\|\\nabla f(x_n)\\|^2 + L^2 \\|x_n - x_\\star\\|^2

    is decreasing along all trajectories and all smooth convex function :math:`f` (i.e., in the worst-case):

    .. math :: V_{n+1} \\leqslant V_n,

    where :math:`x_{n+1}` is obtained from a gradient step from :math:`x_{n}` with fixed step-size :math:`\\gamma=\\frac{1}{L}`.

    **Algorithm**: Onte iteration of radient descent is described by

    .. math:: x_{n+1} = x_n - \\gamma \\nabla f(x_n),

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**:
    The theoretical guarantee can be found in [1, Theorem 3]:

    .. math:: V_{n+1} - V_n \\leqslant 0,

    when :math:`\\gamma=\\frac{1}{L}`.

    **References**: The detailed potential function and SDP approach can be found in [1].

    `[1] A. Taylor, F. Bach (2019). Stochastic first-order methods: non-asymptotic and computer-aided analyses
    via potential functions. Conference on Learning Theory (COLT).
    <https://arxiv.org/pdf/1902.00947.pdf>`_

    Args:
        L (float): the smoothness parameter.
        gamma (float): the step-size.
        n (int):  current iteration number.
        verbose (int): Level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (float): theoretical value.

    Examples:
        >>> L = 1
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_lyapunov_2(L=L, gamma=1 / L, n=10, verbose=1)
        (PEPit) Setting up the problem: size of the main PSD matrix: 4x4
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (0 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                         function 1 : Adding 6 scalar constraint(s) ...
                         function 1 : 6 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 1.894425729310791e-17
        *** Example file: worst-case performance of gradient descent with fixed step size for a given Lyapunov function***
                PEPit guarantee:        V_(n+1) - V_(n) <= 1.89443e-17
                Theoretical guarantee:  V_(n+1) - V_(n) <= 0.0

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

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
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    if gamma == 1 / L:
        theoretical_tau = 0.
    else:
        theoretical_tau = None

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file:'
              ' worst-case performance of gradient descent with fixed step size for a given Lyapunov function***')
        print('\tPEPit guarantee:\t'
              'V_(n+1) - V_(n) <= {:.6}'.format(pepit_tau))
        if gamma == 1 / L:
            print('\tTheoretical guarantee:\t'
                  'V_(n+1) - V_(n) <= {:.6}'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    pepit_tau, theoretical_tau = wc_gradient_descent_lyapunov_2(L=L, gamma=1 / L, n=10, verbose=1)
