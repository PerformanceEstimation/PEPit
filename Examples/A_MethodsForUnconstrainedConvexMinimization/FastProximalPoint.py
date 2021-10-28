import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_fppa(A0, gammas, n, verbose=True):
    """
    Consider the minimization problem
        f_* = min_x f(x),
    where f is convex (possibly non-smooth).

    This code computes a worst-case guarantee for the fast proximal point method. That is, it computes
    the smallest possible tau(n) such that the guarantee
        f(x_n) - f_* <= tau(n) * (f(x_0) - f_* + A/2* ||x_0 - x_*||^2)
    is valid, where x_n is the output of the fast proximal point method, and where x_* is the minimizer of f.

    [1] O. Güler. New proximal point algorithms for convex minimization.
        SIAM Journal on Optimization, 2(4):649–664, 1992.

    :param A0: (float) intial value of A0.
    :param gammas: (list) step size.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function
    func = problem.declare_function(ConvexFunction, {})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is a well-chosen distance between x0 and x^*
    problem.set_initial_condition(func.value(x0) - fs + A0 / 2 * (x0 - xs) ** 2 <= 1)

    # Run the fast proximal point method
    x, v = x0, x0
    A = A0
    for i in range(n):
        alpha = (np.sqrt((A * gammas[i]) ** 2 + 4 * A * gammas[i]) - A * gammas[i]) / 2
        y = (1 - alpha) * x + alpha * v
        x, _, _ = proximal_step(y, func, gammas[i])
        v = v + 1 / alpha * (x - y)
        A = (1 - alpha) * A

    # Set the performance metric to the final distance to optimum in function values
    problem.set_performance_metric(func.value(x) - fs)

    # Solve the PEP
    pepit_tau = problem.solve()

    # Compute theoretical guarantee (for comparison)
    accumulation = 0
    for i in range(n):
        accumulation += np.sqrt(gammas[i])
    theoretical_tau = 4 / A0 / accumulation ** 2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of fast proximal point method ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 3
    A0 = 5
    gammas = [(i + 1) / 1.1 for i in range(n)]

    wc = wc_fppa(A0=A0,
                 gammas=gammas,
                 n=n)
