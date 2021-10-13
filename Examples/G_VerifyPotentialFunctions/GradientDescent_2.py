import cvxpy as cp
import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_convex_function import SmoothConvexFunction



def wc_gd_lyapunov_2(L, gamma, n, verbose=True):
    """
    Consider the minimization problem
        f_* = min_x f(x),
    where f is assumed L-smooth, and x_* = argmin f(x) denotes the minimizer of f.

    This code computes a worst-case guarantee for the gradient descent with fixed steps,
    for a well-chosen Lyapunov function :
        v(x_k) = (2k + 1)*L*(f(x_{k+1}) - f_*) + L/2 || x_k - x_*||^2 + k*(k+2)* ||f'(x_{k+1})||^2
    That is, it computes the smallest possible tau(n, L, mu) such that the guarantee
        v(x_{k+1}) <= tau(n, L) * v(x_k)
    is valid, where x_k is the output of the gradient descent with fixed step size.

    The detailed potential approach and the SDP approach are available in in :
    [1] Adrien Taylor, and Francis Bach. "Stochastic first-order
         methods: non-asymptotic and computer-aided analyses via
         potential functions." (2019)

    :param L: (float) the smoothness parameter.
    :param gamma: (float) the step size.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction,
                                    {'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value gn and fn
    xn = problem.set_initial_point()
    gn, fn = func.oracle(xn)

    # Run the GD at iteration (n+1)
    xnp1 = xn - gamma * gn
    gnp1, fnp1 = func.oracle(xnp1)

    # Compute the Lyapunov function at iteration n and at iteration n+1
    final_lyapunov = L * (2 * n + 1) * (fnp1 - fs) + L**2 * (xnp1 - xs)**2 + n * (n + 2) * gnp1 ** 2
    init_lyapunov = L * (2 * n - 1) * (fn - fs) + L**2 * (xn - xs)**2 + (n - 1) * (n + 1) * gn ** 2

    # Set the initial condition to the bounded initial Lyapunov iterate
    problem.set_initial_condition(init_lyapunov <= 1.)

    # Set the performance metric to the difference between the initial and the final Lyapunov
    problem.set_performance_metric(final_lyapunov)

    # Solve the PEP
    pepit_tau = problem.solve(solver=cp.MOSEK, verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 1.

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of gradient descent with fixed step size for a given Lyapunov function***')
        print('\tPEP-it guarantee:\t\tL(2 n+1)*(f(x_(n+1)) - f_*) + L/2 || x_(n+1) - x_*||^2 + n(n+2) || f\'(x(n+1))||^2  <= {:.6} L(2n - 1)*[(f(x_n) - f_*) + L/2 || x_n - x_*||^2 + (n+1)(n-1) || f\'(x(n))||^2 ] '.format(pepit_tau))
        print('\tTheoretical guarantee:\t L(2n+1)*(f(x_(n+1)) - f_*) + L/2 || x_(n+1) - x_*||^2 + n(n+2) || f\'(x(n+1))||^2 <= {:.6} [L(2n - 1)*(f(x_n) - f_*) + L/2 || x_n - x_*||^2  + (n+1)(n-1) || f\'(x(n))||^2 ]'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 10
    L = 1
    gamma = 1/L

    pepit_tau, theoretical_tau = wc_gd_lyapunov_2(L=L,
                                                    gamma=gamma,
                                                    n=n)