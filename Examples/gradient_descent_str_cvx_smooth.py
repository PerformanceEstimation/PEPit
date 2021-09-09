from math import sqrt

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def compute_rate_n_steps_gd_on_str_cvx_smooth(mu, L, n, alternating_steps=False):
    """
    Compute the rate of n steps of GD methods over strongly convex and smooth functions class.
    Computed either for optimized constant step size or optimized alternating one.

    :param mu: (float) the strong convexity constant.
    :param L: (float) the smoothness constant.
    :param n: (int) number of iterations. Actually, 2.[n/2]is the real number of steps.
    :param alternating_steps: (bool) whether to use alternating step-sizes or not.
    :return:
    """

    # Instantiate PEP
    problem = PEP()

    # Define step sizes, alternating or not
    if alternating_steps:
        gamma1 = 1 / L * (sqrt(L ** 2 + (L - mu) ** 2) - mu) / (L - mu)
        gamma2 = 1 / L * (sqrt(L ** 2 + (L - mu) ** 2) + 2 * L + mu) / (L + 3 * mu)
    else:
        gamma = 2 / (L + mu)
        gamma1 = gamma
        gamma2 = gamma

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction,
                                    {'mu': mu, 'L': L})

    # Start by defining its unique optimal point
    xs = func.optimal_point()

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the GD method
    x = x0
    for _ in range(n // 2):
        x = x - gamma1 * func.gradient(x)
        x = x - gamma2 * func.gradient(x)

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric((x - xs) ** 2)

    # Solve the PEP
    rate = problem.solve()

    # Return the rate of the evaluated method
    return rate


if __name__ == "__main__":

    n = 4
    mu = .1
    L = 1

    rate = compute_rate_n_steps_gd_on_str_cvx_smooth(mu=mu, L=L, n=n, alternating_steps=False)
    accelerated_rate = compute_rate_n_steps_gd_on_str_cvx_smooth(mu=mu, L=L, n=n, alternating_steps=True)

    print('{} < {}'.format(accelerated_rate, rate))
