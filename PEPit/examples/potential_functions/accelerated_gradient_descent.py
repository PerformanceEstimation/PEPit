import numpy as np

from PEPit.pep import PEP
from PEPit.functions.smooth_convex_function import SmoothConvexFunction


def wc_accelerated_gradient_descent(L, gamma, lam, verbose=True):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code computes a worst-case guarantee for **accelerated gradient descent** with fixed step-size :math:`\\gamma`,
    for a well-chosen Lyapunov function:

    .. math:: V_t = \\lambda_t^2 (f(x_t) - f_\\star) + \\frac{L}{2} \\|z_t - x_\\star\\|^2

    That is, it verifies that the above Lyapunov is decreasing on the trajectory:

    .. math :: V_{t+1} \\leq V_t

    is valid, where :math:`x_t`, :math:`z_t`, and :math:`\\lambda_t`
    are defined by the following method.

    **Algorithm**:
    Accelerated gradient descent is described by

    .. math::

        \\begin{eqnarray}
            \\lambda_{t+1} & = & \\frac{1}{2} \\left(1 + \\sqrt{4\\lambda_t^2 + 1}\\right) \\\\
            \\tau_t & = & \\frac{1}{\\lambda_{t+1}} \\\\
            y_t & = & (1 - \\tau_t) x_t + \\tau_t z_t \\\\
            \\eta_t & = & \\frac{\\lambda_{t+1}^2 - \\lambda_{t}^2}{L} \\\\
            z_{t+1} & = & z_t - \\eta_t \\nabla f(y_t) \\\\
            x_{t+1} & = & y_t - \\gamma \\nabla f(y_t)
        \\end{eqnarray}

    **Theoretical guarantee**:
    The theoretical guarantee can be found in [1, Theorem 5.3]:

    .. math:: V_{t+1} \\leq V_t.

    References:

        The detailed potential approach is available in [1, Theorem 5.3].

        `[1] Nikhil Bansal, and Anupam Gupta. "Potential-function proofs for first-order methods." (2019)
        <https://arxiv.org/pdf/1712.04581.pdf>`_

    Args:
        L (float): the smoothness parameter.
        gamma (float): the step-size.
        lam (float): the initial value for sequence :math:`(\\lambda_t)_t`.
        verbose (bool): if True, print conclusion.

    Returns:
        tuple: worst-case value, theoretical value

    Examples:
        >>> L = 1
        >>> pepit_tau, theoretical_tau = wc_accelerated_gradient_descent(L=L, gamma=1 / L, lam=10., verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 6x6
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (0 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 12 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 5.264872499157039e-14
        *** Example file: worst-case performance of accelerated gradient descent for a given Lyapunov function***
            PEP-it guarantee:		[lambda_(t+1)^2 * (f(x_(t+1)) - f_*) + L / 2 ||z_(t+1) - x_*||^2] - [lambda_t^2 * (f(x_t) - f_*) + L / 2 ||z_t - x_*||^2] <= 5.26487e-14
            Theoretical guarantee:	[lambda_(t+1)^2 * (f(x_(t+1)) - f_*) + L / 2 ||z_(t+1) - x_*||^2] - [lambda_t^2 * (f(x_t) - f_*) + L / 2 ||z_t - x_*||^2] <= 0.0

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
    zn = problem.set_initial_point()

    # Run the AGD at iteration (n+1)
    lam_np1 = (1 + np.sqrt(4 * lam ** 2 + 1)) / 2

    tau = 1 / lam_np1
    yn = (1 - tau) * xn + tau * zn
    gyn = func.gradient(yn)

    eta = (lam_np1 ** 2 - lam ** 2) / L
    znp1 = zn - eta * gyn

    xnp1 = yn - gamma * gyn
    gnp1, fnp1 = func.oracle(xnp1)

    # Compute the Lyapunov function at iteration n and at iteration n+1
    final_lyapunov = lam_np1 ** 2 * (fnp1 - fs) + L / 2 * (znp1 - xs) ** 2
    init_lyapunov = lam ** 2 * (fn - fs) + L / 2 * (zn - xs) ** 2

    # Set the performance metric to the difference between the initial and the final Lyapunov
    problem.set_performance_metric(final_lyapunov - init_lyapunov)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 0.

    # Print conclusion if required
    if verbose:
        print(
            '*** Example file: worst-case performance of accelerated gradient descent for a given Lyapunov function***')
        print('\tPEP-it guarantee:\t\t'
              '[lambda_(t+1)^2 * (f(x_(t+1)) - f_*) + L / 2 ||z_(t+1) - x_*||^2]'
              ' - '
              '[lambda_t^2 * (f(x_t) - f_*) + L / 2 ||z_t - x_*||^2] '
              '<= {:.6}'.format(pepit_tau))
        print('\tTheoretical guarantee:\t'
              '[lambda_(t+1)^2 * (f(x_(t+1)) - f_*) + L / 2 ||z_(t+1) - x_*||^2]'
              ' - '
              '[lambda_t^2 * (f(x_t) - f_*) + L / 2 ||z_t - x_*||^2] '
              '<= {:.6}'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    L = 1
    pepit_tau, theoretical_tau = wc_accelerated_gradient_descent(L=L, gamma=1 / L, lam=10., verbose=True)