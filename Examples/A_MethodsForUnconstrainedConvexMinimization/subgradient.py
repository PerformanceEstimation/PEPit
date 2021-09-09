import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.convex_lipschitz_function import ConvexLipschitzFunction


def wc_subgd(M, N, gamma):
    """
    In this example, we use a subgradient method for solving the non-smooth convex minimization problem
    min_x F(x).
    For notational convenience we denote xs=argmin_x F(x), where F(x) satisfies a Lipschitz condition; i.e.,
    it has a bounded gradient ||g||<=M for all g being a subgradient of F at some point.

    We show how to compute the worst-case value of min_i F(xi)-F(xs) when xi is
    obtained by doing i steps of a subgradient method starting with an initial
    iterate satisfying ||x0-xs||<=1.

    :param M: (float) the lipschitz parameter.
    :param N: (int) the number of iterations
    :param gamma: optimal step size is 1/(sqrt(N)*M)
    :return:
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex lipschitz function
    func = problem.declare_function(ConvexLipschitzFunction,
                                    {'M': M})

    # Start by defining its unique optimal point and its function value
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the subgradient method
    x = x0

    for _ in range(N):
        problem.set_performance_metric(func.value(x) - fs)
        x = x - gamma * func.gradient(x)
        _, _ = func.oracle(x)

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric(func.value(x)-fs)

    # Solve the PEP
    wc = problem.solve()

    # Theoretical guarantee (for comparison)
    theory = M/np.sqrt(N+1)

    print('*** Example file: worst-case performance of the subgradient gradient method (OGM) in function values ***')
    print('\tPEP-it guarantee:\t f(y_n)-f_* <= ', wc)
    print('\tTheoretical guarantee:\t f(y_n)-f_* <= ', theory)
    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)

    # Return the rate of the evaluated method
    return wc


if __name__ == "__main__":

    M = 2
    # N = 8 # does not work (eval_points_and_function_values error)
    N = 6 # work
    gamma = 1/(np.sqrt(N+1)*M)

    rate = wc_subgd(M=M,
                    N=N,
                    gamma=gamma)