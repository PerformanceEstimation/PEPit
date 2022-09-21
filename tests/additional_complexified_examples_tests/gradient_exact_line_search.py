import numpy as np

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.point import Point


def wc_gradient_exact_line_search_complexified(L, mu, n, verbose=1):
    """
    See description in `PEPit/examples/unconstrained_convex_minimization/gradient_exact_line_search.py`.
    This example is for testing purposes; the worst-case result is supposed to be the same as that of the other routine,
    but the exact line-search steps are imposed via LMI constraints (instead of relying on the `exact_linesearch_step`
    primitive).

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        n (int): number of iterations.
        verbose (int): Level of information details to print.
                       -1: No verbose at all.
                       0: This example's output.
                       1: This example's output + PEPit information.
                       2: This example's output + PEPit information + CVXPY details.
    
    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_gradient_exact_line_search_complexified(L=1, mu=.1, n=2, verbose=1)
        (PEPit) Setting up the problem: size of the main PSD matrix: 7x7
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: 2 lmi constraint(s) added
                         Size of PSD matrix 1: 2x2
                         Size of PSD matrix 2: 2x2
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                         function 1 : Adding 14 scalar constraint(s) ...
                         function 1 : 14 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.4480925747748075
        *** Example file: worst-case performance of gradient descent with exact linesearch (ELS) ***
                PEPit guarantee:         f(x_n)-f_* <= 0.448093 (f(x_0)-f_*)
                Theoretical guarantee:   f(x_n)-f_* <= 0.448125 (f(x_0)-f_*)

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value g0 and f0
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Set the initial constraint that is the difference between f0 and f_*
    problem.set_initial_condition(f0 - fs <= 1)

    # Run n steps of GD method with ELS
    x = x0
    gx = g0
    for i in range(n):
        gx_prev = gx
        x_prev = x
        x = Point()
        gx, fx = func.oracle(x)

        matrix_of_expressions = np.array([[0, gx_prev * gx], [gx_prev * gx, 0]])
        problem.add_psd_matrix(matrix_of_expressions=matrix_of_expressions)
        func.add_constraint((x - x_prev) * gx == 0)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = ((L - mu) / (L + mu)) ** (2 * n)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent with exact linesearch (ELS) ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_gradient_exact_line_search_complexified(L=1, mu=.1, n=2, verbose=1)
