from PEPit.pep import PEP
from PEPit.Operator_classes.Lipschitz import LipschitzOperator
from PEPit.Primitive_steps.fixedpoint import fixedpoint


def wc_halpern(L, n, verbose=True):
    """
    Consider the fixed point problem
        Find x such that x = Ax,
    where A is a non-expansive operator..

    This code computes a worst-case guarantee for the Halpern Iteration. That is, it computes
    the smallest possible tau(n, L) such that the guarantee
        || x_n - Ax_n||^2 <= tau(n, L) * ||x_0 - x_*||^2
    is valid, where x_n is the output of the Halpern iteration, and x_* the fixed point of A.

    The detailed approach and the tight upper bound are availaible in [1, Theorem 2.1].
    [1] Lieder, Felix. "On the Convergence Rate of the Halpern-Iteration." (2017)

    :param L: (float) the Lipschitz parameter.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a lipschitz operator
    A = problem.declare_function(LipschitzOperator, param={'L': L})

    # Start by defining its unique optimal point xs = x_*
    xs, _, _ = fixedpoint(A)

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
    L = 1

    pepit_tau, theoretical_tau = wc_halpern(L=L,
                                            n=n)
