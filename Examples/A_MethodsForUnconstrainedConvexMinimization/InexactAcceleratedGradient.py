from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Primitive_steps.inexactgradient import inexactgradient


def wc_InexactAGM(L, epsilon, n):
    """
    Consider the convex minimization problem
        f_* = min_x f(x),
    where f is L-smooth and convex.
    This code computes a worst-case guarantee for an accelerated gradient method using inexact first-order information.
    That is, the code computes the smallest possible tau(n,L,epsilon) such that the guarantee
        f(x_n) - f_* <= tau(n,L,epsilon) * ||x_0-x_*||^2
    is valid, where x_n is the output of the inexact accelerated gradient descent, and where x_* is a minimizer of f.
    The inexact descent direction is assumed to satisfy a relative inaccuracy
    described by (with 0 <= epsilon <= 1 )
        || f'(x_i) - d || <= epsilon * || f'(x_i) ||,
    where f'(x_i) is the true gradient, and d is the approximate descent direction that is used.

    In short, for given values of n and L, tau(n,L) is be computed as the worst-case value of f(x_n)-f_* when
    ||x_0-x_*|| == 1.

    :param L: (float) smoothness parameter.
    :param epsilon: (float) level of inaccuracy
    :param n: (int) number of iterations.
    :return:
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, {'mu': 0, 'L': L})

    # Start by defining its unique optimal point
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the accelerated method
    x_new = x0
    y = x0
    for i in range(n):
        dy, fy = inexactgradient(y, func, epsilon, notion='relative')
        x_old = x_new
        x_new = y - 1 / L * dy
        y = x_new + i / (i + 3) * (x_new - x_old)

    # Set the performance metric to the function value accuracy
    _, fx = func.oracle(x_new)
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    wc = problem.solve()
    # Theoretical guarantee (for comparison)
    theory = 2 / (n ** 2 + 5 * n + 6)

    print('*** Example file: worst-case performance of an inexact accelerated gradient method ***')
    print('\tPEP-it guarantee:\t\t\t\t f(x_n)-f_* <= ', wc)
    print('\tKnown upper bound for epsilon=0 (exact case):\t f(x_n)-f_* <= ', theory)
    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return wc, theory


if __name__ == "__main__":
    n = 5
    L = 1
    epsilon = .1

    wc, theory = wc_InexactAGM(L=L, epsilon=epsilon, n=n)

    print('{}'.format(wc))
    print('{}'.format(theory))
