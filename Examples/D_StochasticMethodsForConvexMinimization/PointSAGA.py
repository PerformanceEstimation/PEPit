import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_psaga(L, mu, n, verbose=True):
    """
    Consider the finite sum minimization problem
        f_* = min_x {F(x) = 1/n [f1(x) + ... + fn(x)]},
    where f1, ..., fn are assumed L-smooth and mu-strongly convex,
    and with proximal operator available.

    This code computes the exact rate for the Lyapunov function from the original Point SAGA paper,
    given in Theorem 5 of [1].

    [1] Aaron Defazio, Francis Bach, and Simon Lacoste-Julien.
        "SAGA: A fast incremental gradient method with support for
        non-strongly convex composite objectives." (2014)
        (Theorem 1 of [1])

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a sum of strongly convex functions
    fn = [problem.declare_function(SmoothStronglyConvexFunction, param={'L': L, 'mu': mu}) for _ in range(n)]
    func = np.mean(fn)

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the initial values
    phi = [problem.set_initial_point() for _ in range(n)]
    x0 = problem.set_initial_point()

    # Parameters of the scheme and of the Lyapunov function
    gamma = np.sqrt((n - 1) ** 2 + 4 * n * L / mu) / 2 / L / n - (1 - 1 / n) / 2 / L
    c = 1 / mu / L

    # Compute the initial value of the Lyapunov function
    T0 = (xs - x0) ** 2
    for i in range(n):
        gis, fis = fn[i].oracle(xs)
        T0 = T0 + c / n * (gis - phi[i]) ** 2

    # Set the initial constraint as the Lyapunov bounded by 1
    problem.set_initial_condition(T0 <= 1.)

    # Compute the expected value of the Lyapunov function after one iteration
    # (so: expectation over n possible scenarios:  one for each element fi in the function).
    T1avg = (xs - xs) ** 2
    for i in range(n):
        w = x0 + gamma * phi[i]
        for j in range(n):
            w = w - gamma / n * phi[j]
        x1, gx1, _ = proximal_step(w, fn[i], gamma)
        T1 = (xs - x1) ** 2
        for j in range(n):
            gjs, fjs = fn[j].oracle(xs)
            if i != j:
                T1 = T1 + c / n * (phi[j] - gjs) ** 2
            else:
                T1 = T1 + c / n * (gjs - gx1) ** 2
        T1avg = T1avg + T1 / n

    # Set the performance metric to the distance average to optimal point
    problem.set_performance_metric(T1avg)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison) : the bound is given in theorem 5 of [1]
    kappa = mu * gamma / (1 + mu * gamma)
    theoretical_tau = (1 - kappa)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of Point SAGA for a given Lyapunov function ***')
        print('\tPEP-it guarantee:\t\t T1(x_n, x_*) <= {:.6} TO(x_n, x_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t T1(x_n, x_*) <= {:.6} TO(x_n, x_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 10
    L = 1
    mu = 0.1

    pepit_tau, theoretical_tau = wc_psaga(L=L,
                                          mu=mu,
                                          n=n)
