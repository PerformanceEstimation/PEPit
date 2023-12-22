from math import sqrt

from PEPit import PEP
from PEPit.point import Point
from PEPit.operators import NonexpansiveOperator


def wc_inconsistent_halpern_iteration(n, verbose=1):
    """
    Consider the fixed point problem

    .. math:: \\mathrm{Find}\\, x:\\, x = Ax,

    where :math:`A` is a non-expansive operator,
    that is a :math:`L`-Lipschitz operator with :math:`L=1`.
    When the solution of above problem, or fixed point, does not exist,
    behavior of the fixed-point iteration with A can be characterized with
    infimal displacement vector :math:`v`.

    This code computes a worst-case guarantee for the **Halpern Iteration**,
    when `A` is not necessarily consistent, i.e., does not necessarily have fixed point.
    That is, it computes the smallest possible :math:`\\tau(n)` such that the guarantee

    .. math:: \\|x_n - Ax_n - v\\|^2 \\leqslant \\tau(n) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the **Halpern iteration**
    and :math:`x_\\star` is the point where :math:`v` is attained, i.e.,

    .. math:: v = x_\\star - Ax_\\star

    In short, for a given value of :math:`n`,
    :math:`\\tau(n)` is computed as the worst-case value of
    :math:`\\|x_n - Ax_n - v\\|^2` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**: The Halpern iteration can be written as

        .. math:: x_{t+1} = \\frac{1}{t + 2} x_0 + \\left(1 - \\frac{1}{t + 2}\\right) Ax_t.

    **Theoretical guarantee**: A worst-case guarantee for Halpern iteration can be found in [1, Theorem 2.1]:

        .. math:: \\|x_n - Ax_n - v\\|^2 \\leqslant \\left(\\frac{2}{n+1}\\right)^2 \\|x_0 - x_\\star\\|^2.

    **References**: The detailed approach is available in [1].

    `[1] J. Park, E. Ryu (2023). Accelerated Infeasibility Detection of Constrained Optimization and Fixed-Point Iterations. International Conference on Machine Learning.
    <http://https://arxiv.org/abs/2303.15876>`_

    Args:
        n (int): number of iterations.
        verbose (int): Level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_inconsistent_halpern_iteration(n=25, verbose=1)
        (PEPit) Setting up the problem: size of the main PSD matrix: 29x29
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                         function 1 : Adding 729 scalar constraint(s) ...
                         function 1 : 729 scalar constraint(s) added
        (PEPit) Setting up the problem: constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); optimal value: 0.026779232124681585
        *** Example file: worst-case performance of Halpern Iterations ***
                PEPit guarantee:         ||xN - AxN - v||^2 <= 0.0267792 ||x0 - x_*||^2
                Theoretical guarantee:   ||xN - AxN - v||^2 <= 0.0213127 ||x0 - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a non expansive operator
    A = problem.declare_function(NonexpansiveOperator)

    # Start by defining point xs where infimal displacement vector v is attained
    xs = Point()
    Txs = A.gradient(xs)
    A.v = xs - Txs

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the difference between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of Halpern Iterations
    x = x0
    for i in range(n):
        x = 1 / (i + 2) * x0 + (1 - 1 / (i + 2)) * A.gradient(x)

    # Set the performance metric to distance between xN - AxN and v
    problem.set_performance_metric((x - A.gradient(x) - A.v) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)
    
    # Compute theoretical guarantee (for comparison)
    sum = 0
    for cnt in range(n):
        sum += 1 / (cnt + 1)
    theoretical_tau = ( (sqrt(sum + 4) + 1) / (n + 1) ) ** 2

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of (possibly inconsistent) Halpern Iterations ***')
        print('\tPEPit guarantee:\t ||xN - AxN - v||^2 <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||xN - AxN - v||^2 <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_inconsistent_halpern_iteration(n=25, verbose=1)
    

