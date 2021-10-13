import cvxpy as cp

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_convex_function import SmoothConvexFunction



def wc_gd_lyapunov_1(L, gamma, n, verbose=True):
    """
    Consider the minimization problem
        f_* = min_x f(x),
    where f is assumed L-smooth and x_* = argmin f(x) denotes the minimizer of f.

    This code computes a worst-case guarantee for the gradient descent with fixed steps,
    for a well-chosen Lyapunov function :
        v(x_k) = (k)*(f(x_{k}) - f_*) + L/2 || x_k - x_*||^2
    That is, it computes the smallest possible tau(n, L, mu) such that the guarantee
        v(x_{k+1}) <= tau(n, L, mu) * v(x_k)
    is valid, where x_k is the output of the gradient descent with fixed step size.

    The detailed potential approach is available in [1, Theorem 3.3], and the SDP approach in [2].
    [1] Nikhil Bansal, and Anupam Gupta.  "Potential-function proofs for
         first-order methods." (2019)

    [2] Adrien Taylor, and Francis Bach. "Stochastic first-order
         methods: non-asymptotic and computer-aided analyses via
         potential functions." (2019)

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param gamma: (float) the step size.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothConvexFunction,
                                    {'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value gn and fn
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Run the GD at iteration (n+1)
    x1 = x0 - gamma * g0
    g1, f1 = func.oracle(x1)

    # Compute the Lyapunov function at iteration n and at iteration n+1
    final_lyapunov = (n + 1) * (f1 - fs) + L/2 * (x1 - xs)**2
    init_lyapunov = n * (f0 - fs) + L/2 * (x0 - xs)**2

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
        print('\tPEP-it guarantee:\t\t(n+1)*(f(x_(n+1)) - f_*) + L/2 || x_(n+1) - x_*||^2  <= {:.6} (n)*(f(x_n) - f_*) + L/2 || x_n - x_*||^2 '.format(pepit_tau))
        print('\tTheoretical guarantee:\t (n+1)*(f(x_(n+1)) - f_*) + L/2 || x_(n+1) - x_*||^2  <= {:.6} (n)*(f(x_n) - f_*) + L/2 || x_n - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 10
    L = 1
    gamma = 1/L

    pepit_tau, theoretical_tau = wc_gd_lyapunov_1(L=L,
                                                    gamma=gamma,
                                                    n=n)