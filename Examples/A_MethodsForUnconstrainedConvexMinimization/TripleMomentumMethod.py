import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_tmm(mu, L, n):
    """
    In this example, we use the triple momentum method for solving the
    L-smooth mu-strongly convex minimization problem
       min_x F(x);
    for notational convenience we denote xs=argmin_x F(x).

    We show how to compute the worst-case value of F(xN)-F(xs) when xN is
    obtained by doing N steps of the method starting with an initial
    iterate satisfying ||x0-xs||<=1.

    [1] Van Scoy, B., Freeman, R. A., & Lynch, K. M. (2018).
    "The fastest known globally convergent first-order method for
    minimizing strongly convex functions."
    IEEE Control Systems Letters, 2(1), 49-54.

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param n: (int) number of iterations.
    :return:
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex lipschitz function
    func = problem.declare_function(SmoothStronglyConvexFunction,
                                    {'mu': mu, 'L': L})

    # Start by defining its unique optimal point and its function value
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # algorithmic parameters
    kappa = L / mu
    rho = (1 - 1 / np.sqrt(kappa))
    alpha = (1 + rho) / L
    beta = rho ** 2 / (2 - rho)
    gamma = rho ** 2 / (1 + rho) / (2 - rho)
    delta = rho ** 2 / (1 - rho ** 2)

    # Run the triple momentum method
    x_old = x0
    x_new = x0
    y = x0

    for _ in range(n + 1):
        x_inter = (1 + beta) * x_new - beta * x_old - alpha * func.gradient(y)
        y = (1 + gamma) * x_inter - gamma * x_new
        x = (1 + delta) * x_inter - delta * x_new
        x_new, x_old = x_inter, x_new

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric(func.value(x) - fs)

    # Solve the PEP
    wc = problem.solve()

    # Theoretical guarantee (for comparison)
    theory = rho ** (2 * (n + 1)) * L / 2 * kappa
    print('*** Example file: worst-case performance of the triple momentum method (TMM) in function values ***')
    print('\tPEP-it guarantee:\t f(y_n)-f_* <= ', wc)
    print('\tTheoretical guarantee for L/mu large :\t f(y_n)-f_* <= ', theory)
    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)

    # Return the rate of the evaluated method
    return wc


if __name__ == "__main__":
    mu = 0.1
    L = 1.
    n = 4

    rate = wc_tmm(mu=mu,
                  L=L,
                  n=n)
