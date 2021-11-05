from PEPit.pep import PEP
from PEPit.operators.lipschitz import LipschitzOperator
from PEPit.primitive_steps.fixed_point import fixed_point


def wc_halpern(n, verbose=True):
    """
    Consider the fixed point problem
        Find x such that x = Ax,
    where A is a non-expansive operator., that is a L-Lipschitz operator with L=1.

    This code computes a worst-case guarantee for the Halpern Iteration. That is, it computes
    the smallest possible tau(n) such that the guarantee
        || x_n - Ax_n||^2 <= tau(n) * ||x_0 - x_*||^2
    is valid, where x_n is the output of the Halpern iteration, and x_* the fixed point of A.

    The detailed approach and the tight upper bound are available in [1, Theorem 2.1].
    [1] Lieder, Felix. "On the Convergence Rate of the Halpern-Iteration." (2017)

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

    # Run n steps of Halpern Iterations
    x = x0
    for i in range(n):
        x = 1 / (i + 2) * x0 + (1 - 1 / (i + 2)) * A.gradient(x)
    Ax = A.gradient(x)

    # Set the performance metric to distance between xN and AxN
    problem.set_performance_metric((x - Ax) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (2 / (n + 1)) ** 2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of Halpern Iterations ***')
        print('\tPEP-it guarantee:\t\t || xN - AxN ||^2 <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t || xN - AxN ||^2 <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 10

    pepit_tau, theoretical_tau = wc_halpern(n=n)
