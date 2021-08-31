from PEPit.pep import PEP
from PEPit.Function_classes.strong_cvx_smooth_function import StrongCvxSmoothFunction


def wc_gd(mu, L, gamma, n):
    """
    This code computes the worst-case convergence rate of gradient descent towards a minimizer of a
    L-smooth m-strongly convex function, f (hence, x_* is unique if m>0).
    That is, we compute the smallest value of "rate" such that the inequality
    ||x_{n} - x_* ||^2 <= rate * || x_0 - x_* ||^2
    is valid for all x_0 and x_*, and all L-smooth m-strongly convex function f with x_* being a minimizer of f and
    x_{n} being computed as a gradient step; x_{k+1}=x_k - gamma * grad f (x_k) (k=0,...,n-1)

    :param mu: (float) the strong convexity parameter.
    :param L: (float) the smoothness parameter.
    :param gamma: (float) step size.
    :param n: (int) number of iterations.
    :return:
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(StrongCvxSmoothFunction,
                                    {'mu': mu, 'L': L})

    # Start by defining its unique optimal point
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the GD method
    x = x0
    for _ in range(n):
        x = x - gamma * func.gradient(x)

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric(func.value(x)-fs)

    # Solve the PEP
    wc = problem.solve()

    # Return the rate of the evaluated method
    return wc


if __name__ == "__main__":

    n = 2
    mu = 0
    L = 1

    rate = rate_gd(mu=mu, L=L, gamma=1/L, n=n)

    print('{}'.format(rate))
