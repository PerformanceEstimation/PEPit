from math import sqrt
import numpy as np

from PEPit import PEP
from PEPit.point import Point
from PEPit.operators import NonexpansiveOperator


def wc_inconsistent_halpern_iteration(n, wrapper="cvxpy", solver=None, verbose=1):
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

    **Theoretical guarantee**: A worst-case guarantee for Halpern iteration can be found in [1, Theorem 8]:

        .. math:: \\|x_n - Ax_n - v\\|^2 \\leqslant \\left(\\frac{\\sqrt{Hn + 12} + 1}{n + 1}\\right)^2 \\|x_0 - x_\\star\\|^2.

    **References**: The detailed approach is available in [1].

    `[1] J. Park, E. Ryu (2023).
    Accelerated Infeasibility Detection of Constrained Optimization and Fixed-Point Iterations.
    International Conference on Machine Learning.
    <https://arxiv.org/pdf/2303.15876.pdf>`_

    Args:
        n (int): number of iterations.
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
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
        (PEPit) Setting up the problem: size of the Gram matrix: 29x29
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 378 scalar constraint(s) ...
        			Function 1 : 378 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.02678884717170149
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite
        		All the primal scalar constraints are verified up to an error of 4.2928149923682213e-10
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 3.9511359559460285e-06
        (PEPit) Final upper bound (dual): 0.02678881575052497 and lower bound (primal example): 0.02678884717170149 
        (PEPit) Duality gap: absolute: -3.142117652177312e-08 and relative: -1.1729200708183147e-06
        *** Example file: worst-case performance of (possibly inconsistent) Halpern Iterations ***
        	PEPit guarantee:	 ||xN - AxN - v||^2 <= 0.0267888 ||x0 - x_*||^2
        	Theoretical guarantee:	 ||xN - AxN - v||^2 <= 0.0366417 ||x0 - x_*||^2
    
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
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    Hn = np.sum(1 / np.arange(1, n+1))
    theoretical_tau = ((sqrt(Hn + 12) + 1) / (n + 1)) ** 2

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of (possibly inconsistent) Halpern Iterations ***')
        print('\tPEPit guarantee:\t ||xN - AxN - v||^2 <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||xN - AxN - v||^2 <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_inconsistent_halpern_iteration(n=25, wrapper="cvxpy", solver=None, verbose=1)
