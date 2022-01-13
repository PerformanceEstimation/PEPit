from math import sqrt

from PEPit import PEP
from PEPit.functions import SmoothConvexFunction


def wc_accelerated_gradient_method(L, gamma, lam, verbose=True):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code verifies a worst-case guarantee for an **accelerated gradient method**. That is, it verifies
    that the Lyapunov (or potential/energy) function

    .. math:: V_n \\triangleq \\lambda_n^2 (f(x_n) - f_\\star) + \\frac{L}{2} \\|z_n - x_\\star\\|^2

    is decreasing along all trajectories and all smooth convex function :math:`f` (i.e., in the worst-case):

    .. math :: V_{n+1} \\leqslant V_n,

    where :math:`x_{n+1}`, :math:`z_{n+1}`, and :math:`\\lambda_{n+1}` are obtained from one iteration of
    the accelerated gradient method below, from some arbitrary :math:`x_{n}`, :math:`z_{n}`, and :math:`\\lambda_{n}`.

    **Algorithm**: One iteration of accelerated gradient method is described by

    .. math::

        \\begin{eqnarray}
            \\text{Set: }\\lambda_{n+1} & = & \\frac{1}{2} \\left(1 + \\sqrt{4\\lambda_n^2 + 1}\\right), \\tau_n & = & \\frac{1}{\\lambda_{n+1}},
            \\text{ and } \\eta_n & = & \\frac{\\lambda_{n+1}^2 - \\lambda_{n}^2}{L} \\\\
            y_n & = & (1 - \\tau_n) x_n + \\tau_n z_n,\\\\
            z_{n+1} & = & z_n - \\eta_n \\nabla f(y_n), \\\\
            x_{n+1} & = & y_n - \\gamma \\nabla f(y_n).
        \\end{eqnarray}

    **Theoretical guarantee**: The following worst-case guarantee can be found in e.g., [2, Theorem 5.3]:

    .. math:: V_{n+1} - V_n  \\leqslant 0,

    when :math:`\\gamma=\\frac{1}{L}`.

    **References**: The potential can be found in the historical [1]; and in more recent works, e.g., [2, 3].

    `[1] Y. Nesterov (1983). A method for solving the convex programming problem with convergence rate :math:`O(1/k^2)`.
    In Dokl. akad. nauk Sssr (Vol. 269, pp. 543-547).
    <http://www.mathnet.ru/links/9bcb158ed2df3d8db3532aafd551967d/dan46009.pdf>`_

    `[2] N. Bansal, A. Gupta (2019). Potential-function proofs for gradient methods. Theory of Computing, 15(1), 1-32.
    <https://arxiv.org/pdf/1712.04581.pdf>`_

    `[3] A. d’Aspremont, D. Scieur, A. Taylor (2021). Acceleration Methods. Foundations and Trends
    in Optimization: Vol. 5, No. 1-2.
    <https://arxiv.org/pdf/2101.09545.pdf>`_

    Args:
        L (float): the smoothness parameter.
        gamma (float): the step-size.
        lam (float): the initial value for sequence :math:`(\\lambda_t)_t`.
        verbose (bool): if True, print conclusion.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Examples:
        >>> L = 1
        >>> pepit_tau, theoretical_tau = wc_accelerated_gradient_method(L=L, gamma=1 / L, lam=10., verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 6x6
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (0 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 12 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); optimal value: 7.946321396432764e-09
        *** Example file: worst-case performance of accelerated gradient method for a given Lyapunov function***
            PEPit guarantee:       V_(n+1) - V_n <= 7.94632e-09
            Theoretical guarantee:  V_(n+1) - V_n <= 0.0

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
    lam_np1 = (1 + sqrt(4 * lam ** 2 + 1)) / 2

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
    if gamma == 1/L:
        theoretical_tau = 0.
    else:
        theoretical_tau = None

    # Print conclusion if required
    if verbose:
        print(
            '*** Example file: worst-case performance of accelerated gradient method for a given Lyapunov function***')
        print('\tPEPit guarantee:\t V_(n+1) - V_n <= {:.6}'.format(pepit_tau))
        if gamma == 1/L:
            print('\tTheoretical guarantee:\t V_(n+1) - V_n <= {:.6}'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    L = 1
    pepit_tau, theoretical_tau = wc_accelerated_gradient_method(L=L, gamma=1 / L, lam=10., verbose=True)
