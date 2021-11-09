import numpy as np

from PEPit.pep import PEP
from PEPit.operators.lipschitz import LipschitzOperator
from PEPit.primitive_steps.fixed_point import fixed_point


def wc_km(n, verbose=True):
    """
    Consider the fixed point problem
        Find x such that x = Ax,
    where A is a non-expansive operator., that is a L-Lipschitz operator wit L=1.

    This code computes a worst-case guarantee for the Krasnolselskii-Mann. That is, it computes
    the smallest possible tau(n) such that the guarantee
        1/4|| x_n - Ax_n||^2 <= tau(n) * ||x_0 - x_*||^2
    is valid, where x_n is the output of the Krasnolseskii-Mann iterations, and x_* the fixed point of A.

    This scheme was first studied using PEPs in [1, Theorem 4.9], with a theoretical upper bound:
    [1] Felix Lieder. "Projection Based Methods for Conic Linear Programming
        Optimal First Order Complexities and Norm Constrained Quasi Newton
        Methods."  PhD thesis (2018)

    :param L: (float) the Lipschitz parameter.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a non expansive operator
    A = problem.declare_function(LipschitzOperator, param={'L': 1.})

    # Start by defining its unique optimal point xs = x_*
    xs, _, _ = fixed_point(A)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the difference between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    x = x0
    for i in range(n):
        x = 1 / (i + 2) * x + (1 - 1 / (i + 2)) * A.gradient(x)
    Ax = A.gradient(x)

    # Set the performance metric to distance between xN and AxN
    problem.set_performance_metric((1 / 2 * (x - Ax)) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    tn = 1 - 1 / (n + 1)
    if (tn >= 1 / 2) & (tn <= 1 / 2 * (1 + np.sqrt(n / (n + 1)))):
        theoretical_tau = 1 / (n + 1) * (n / (n + 1)) ** n / (4 * tn * (1 - tn))
    if (tn <= 1) & (tn > 1 / 2 * (1 + np.sqrt(n / (n + 1)))):
        theoretical_tau = (2 * tn - 1) ** (2 * n)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of Kranoselskii-Mann iterations ***')
        print('\tPEP-it guarantee:\t\t 1/4|| xN - AxN ||^2 <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t 1/4|| xN - AxN ||^2 <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 3
    pepit_tau, theoretical_tau = wc_km(n=n)
