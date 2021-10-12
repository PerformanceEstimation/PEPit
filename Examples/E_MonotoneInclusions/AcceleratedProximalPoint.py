import cvxpy as cp

from PEPit.pep import PEP
from PEPit.Operator_classes.Monotone import MonotoneOperator
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_ppm(alpha, n, verbose=True):
    """
    Consider the monotone inclusion problem
        Find x, 0 \in Ax,
    where A is maximally monotone. we denote JA the resolvents of A.

    This code computes a worst-case guarantee for the accelerated proximal point method, that is the smallest
    possible tau(n,L) such that the guarantee
        || x_(n) - y(n)||^2 <= tau(n,L) * || x_0 - x_*||^2,
    is valid, where x_* is such that 0 in Ax_*..

    Theoretical rates can be found in the following paper (section 4, Theorem 4.1)

    [1] Donghwan Kim. "Accelerated Proximal Point Method  and Forward Method
        for Monotone Inclusions." (2019)

    :param alpha: (float) the step size
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a monotone operator
    A = problem.declare_function(MonotoneOperator, {})

    # Start by defining its unique optimal point xs = x_*
    xs = A.optimal_point()

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute n steps of the Proximal Gradient method starting from x0
    x = [x0 for _ in range(n+1)]
    y = [x0 for _ in range(n+1)]
    for i in range(0, n-1):
        x[i+1], _, _ = proximal_step(y[i+1], A, alpha)
        y[i+2] = x[i+1] + i / (i + 2) * (x[i+1] - x[i]) - i / (i + 2) * (x[i] - y[i])
    x[n], _, _ = proximal_step(y[n], A, alpha)

    # Set the performance metric to the distance between xn and yn
    problem.set_performance_metric((x[n] - y[n])**2)

    # Solve the PEP
    pepit_tau = problem.solve(cp.MOSEK)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 1/(n)**2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Accelerated Proximal Point Method***')
        print('\tPEP-it guarantee:\t ||x(n) - y(n)||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee :\t ||x(n) - y(n)||^2 <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    alpha = 2
    n = 10

    pepit_tau, theoretical_tau = wc_ppm(alpha=alpha,
                                        n=n)
