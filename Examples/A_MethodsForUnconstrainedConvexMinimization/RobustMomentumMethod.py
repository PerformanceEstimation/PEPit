import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_rmm(mu, L, lam):
    """
    In this example, we use the robust momentum method for solving the
    L-smooth mu-strongly convex minimization problem
        min_x F(x);
    for notational convenience we denote xs=argmin_x F(x).
    We show how to compute the tight rate for the Lyapunov function
    developped in
    [1] Cyrus, S., Hu, B., Van Scoy, B., & Lessard, L. "A robust accelerated
         optimization algorithm for strongly convex functions." In 2018 Annual
         American Control Conference (ACC) (pp. 1376-1381). IEEE.

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param lam: (float) if lam=1 it is the gradient descent, if lam=0, it is the triple momentum method.
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
    x1 = problem.set_initial_point()

    # algorithmic parameters
    kappa = L/mu
    rho = lam*(1 - 1 / kappa) + (1 - lam) * (1 - 1 / float(np.sqrt(kappa)))
    alpha = kappa * (1 - rho)**2*(1 + rho)/L
    beta = kappa * rho**3 / (kappa - 1)
    gamma = rho**3 / ((kappa - 1) * (1 - rho)**2 * (1 + rho))
    l = mu ** 2 * (kappa - kappa * rho**2 - 1) / (2 * rho * (1-rho))
    nnu = (1 + rho) * (1 - kappa + 2 * kappa * rho - kappa * rho**2) / (2 * rho)

    # Run the robust momentum method
    y0 = x1 + gamma * (x1 - x0)
    g0, f0 = func.oracle(y0)
    x2 = x1 + beta * (x1-x0) - alpha * g0
    y1 = x2 + gamma * (x2 - x1)
    g1, f1 = func.oracle(y1)
    x3 = x2 + beta * (x2 - x1) - alpha * g1

    z1 = (x2 - (rho**2) * x1) / (1 - rho**2)
    z2 = (x3 - (rho**2) * x2) / (1 - rho**2)

    # Evaluation the lyapunov function at the first and second iteration
    q0 = (L-mu)*(f0 - fs - mu/2*(y0-xs)**2) - 1/2*(g0 - mu*(y0-xs))**2
    q1 = (L-mu)*(f1 - fs - mu/2*(y1-xs)**2) - 1/2*(g1 - mu*(y1-xs))**2
    initLyapunov = l*(z1 - xs)**2 + q0
    finalLyapunov = l*(z2 - xs)**2 + q1
    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition(initLyapunov <= 1)

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric(finalLyapunov)

    # Solve the PEP
    wc = problem.solve()

    # Theoretical guarantee (for comparison)
    theory = rho**2
    print('*** Example file: worst-case performance of the Robust Momentum Method (RMM) in function values ***')
    print('\tPEP-it guarantee:\t f(y_n)-f_* <= ', wc)
    print('\tTheoretical guarantee :\t f(y_n)-f_* <= ', theory)
    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)

    # Return the rate of the evaluated method
    return wc, theory


if __name__ == "__main__":

    mu = 0.1
    L = 1.
    lam = 0.2

    rate = wc_rmm(mu=mu,
                  L=L,
                  lam=lam)