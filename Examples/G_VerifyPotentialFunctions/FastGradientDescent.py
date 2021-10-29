import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_convex_function import SmoothConvexFunction


def wc_gd_lyapunov(L, gamma, lam, verbose=True):
    """
    Consider the minimization problem
        f_* = min_x f(x),
    where f is assumed L-smooth and x_* = argmin f(x) denotes the minimizer of f.

    This code computes a worst-case guarantee for the accelerated gradient descent,
    for a well-chosen Lyapunov function :
        v(x_{k+1}) = lambda_k^2 (f(x_{k+1}) - f_*) + L/2 * || z_{k+1} - xs||^2
        with lambda_{k+1} = 1/2 * (1 - sqrt(4*lambda_k^2 + 1))
    where :
        y_{k} = (1 - tau_k) x_k + tau_k z_k
        x_{k+1} = y_k - 1/L f'(x_k)
        z_{k+1} = z_k - eta_k f'(y_k)
    with taux_k = 1/lambda_k and eta_k = (lambda_k^2 - lambda_{k-1}^2) / L

    That is, it verifies that the Lyapunov v(.) is decreasing on the trajectory :
        v(x_{k+1}) - v(x_k) <= 0.
    is valid, where x_k is the output of the accelerated gradient descent.

    The detailed potential approach is available in [1, Theorem 5.3].
    [1] Nikhil Bansal, and Anupam Gupta.  "Potential-function proofs for
         first-order methods." (2019)

    :param L: (float) the smoothness parameter.
    :param gamma: (float) the step size.
    :param lam: (float) the initial value for sequence (lambda_k)_k.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction,
                                    param={'L': L})

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
    eta = (lam_np1 ** 2 - lam ** 2) / L

    yn = (1 - tau) * xn + tau * zn
    gyn = func.gradient(yn)
    xnp1 = yn - gamma * gyn
    znp1 = zn - eta * gyn
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
              '(n+1)*(f(x_(n+1)) - f_*) + L/2 || x_(n+1) - x_*||^2'
              ' - '
              '[n*(f(x_n) - f_*) + L/2 || x_n - x_*||^2] '
              '<= {:.6}'.format(pepit_tau))
        print('\tTheoretical guarantee:\t'
              '(n+1)*(f(x_(n+1)) - f_*) + L/2 || x_(n+1) - x_*||^2'
              ' - '
              '[n*(f(x_n) - f_*) + L/2 || x_n - x_*||^2] '
              '<= {:.6}'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    lam = 10.
    gamma = 1 / L

    pepit_tau, theoretical_tau = wc_gd_lyapunov(L=L,
                                                gamma=gamma,
                                                lam=lam)
