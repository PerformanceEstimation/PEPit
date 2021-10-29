import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_heavyball(mu, L, alpha, beta, n, verbose=True):
    """
    Consider the convex minimization problem
        f_* = min_x f(x),
    where f is L-smooth and mu-strongly-convex.

    This code computes a worst-case guarantee for the Heavy-ball method :
    x_{k+1} = x_k - alpha*grad(f(x_k)) + beta*(x_k-x_{k-1})

    That is, it computes the smallest possible tau(n, mu, L) such that the guarantee
        f(x_n) - f_* <= tau(n, mu, L) (f(x_0) -  f(x_*)),
    is valid, where x_n is the output of the heavy-ball method, and where x_* is a minimizer of f.

    This methods was first introduce in [1], and convergence upper bound was proved in [2].
    [1] B.T. Polyak.
    "Some methods of speeding up the convergence of iteration methods".
    [2]  Euhanna Ghadimi, Hamid Reza Feyzmahdavian, Mikael. Johansson.
    " Global convergence of the Heavy-ball method for convex optimization".

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param alpha: (float) parameter of the scheme.
    :param beta: (float) parameter of the scheme such that 0<beta<1 and 0<alpha<2*(1+beta)
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding function value f0
    x0 = problem.set_initial_point()
    f0 = func.value(x0)

    # Set the initial constraint that is the distance between f(x0) and f(x^*)
    problem.set_initial_condition((f0 - fs) <= 1)

    # Run one step of the heavy ball method
    x_old = x0
    g_old = func.gradient(x_old)
    x_new = x_old - alpha * g_old
    g_new, f_new = func.oracle(x_new)

    for _ in range(n):
        x_new, x_old = x_new - alpha * g_new + beta * (x_new - x_old), x_new
        g_new, f_new = func.oracle(x_new)

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric(f_new - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (1 - alpha * mu) ** (n + 1)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the fast gradient method ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} (f(x_0) -  f(x_*))'.format(
            pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0) -  f(x_*))'.format(
            theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    mu = 0.1
    L = 1.

    # Optimal parameters for differentiable functions
    alpha = 1 / (2 * L)  # alpha \in [0, 1/L]
    beta = np.sqrt((1 - alpha * mu) * (1 - L * alpha))
    n = 1

    pepit_tau, theoretical_tau = wc_heavyball(mu=mu,
                                              L=L,
                                              alpha=alpha,
                                              beta=beta,
                                              n=n)
