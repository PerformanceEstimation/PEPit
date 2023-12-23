from PEPit import PEP
from PEPit.operators import LipschitzOperator


def wc_halpern_iteration(n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the fixed point problem

    .. math:: \\mathrm{Find}\\, x:\\, x = Ax,

    where :math:`A` is a non-expansive operator,
    that is a :math:`L`-Lipschitz operator with :math:`L=1`.

    This code computes a worst-case guarantee for the **Halpern Iteration**.
    That is, it computes the smallest possible :math:`\\tau(n)` such that the guarantee

    .. math:: \\|x_n - Ax_n\\|^2 \\leqslant \\tau(n) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the **Halpern iteration**,
    and :math:`x_\\star` the fixed point of :math:`A`.

    In short, for a given value of :math:`n`,
    :math:`\\tau(n)` is computed as the worst-case value of
    :math:`\\|x_n - Ax_n\\|^2` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**: The Halpern iteration can be written as

        .. math:: x_{t+1} = \\frac{1}{t + 2} x_0 + \\left(1 - \\frac{1}{t + 2}\\right) Ax_t.

    **Theoretical guarantee**: A **tight** worst-case guarantee for Halpern iteration can be found in [2, Theorem 2.1]:

        .. math:: \\|x_n - Ax_n\\|^2 \\leqslant \\left(\\frac{2}{n+1}\\right)^2 \\|x_0 - x_\\star\\|^2.

    **References**: The method was first proposed in [1]. The detailed analysis and tight bound are available in [2].

    `[1] B. Halpern (1967).
    Fixed points of nonexpanding maps.
    American Mathematical Society, 73(6), 957–961.
    <https://www.ams.org/journals/bull/1967-73-06/S0002-9904-1967-11864-0/S0002-9904-1967-11864-0.pdf>`_

    `[2] F. Lieder (2021).
    On the convergence rate of the Halpern-iteration.
    Optimization Letters, 15(2), 405-418.
    <http://www.optimization-online.org/DB_FILE/2017/11/6336.pdf>`_

    Args:
        n (int): number of iterations.
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + solver details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_halpern_iteration(n=25, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 28x28
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 351 scalar constraint(s) ...
        			Function 1 : 351 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.005917282090077699
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 3.755174265259971e-09
        		All the primal scalar constraints are verified up to an error of 1.3075734314749177e-08
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 4.4389586421813856e-09
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 2.425619306975869e-07
        (PEPit) Final upper bound (dual): 0.005917288963138354 and lower bound (primal example): 0.005917282090077699 
        (PEPit) Duality gap: absolute: 6.873060655332441e-09 and relative: 1.1615232383221049e-06
        *** Example file: worst-case performance of Halpern Iterations ***
        	PEPit guarantee:	 ||xN - AxN||^2 <= 0.00591729 ||x0 - x_*||^2
        	Theoretical guarantee:	 ||xN - AxN||^2 <= 0.00591716 ||x0 - x_*||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a non expansive operator
    A = problem.declare_function(LipschitzOperator, L=1.)

    # Start by defining its unique optimal point xs = x_*
    xs, _, _ = A.fixed_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the difference between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of Halpern Iterations
    x = x0
    for i in range(n):
        x = 1 / (i + 2) * x0 + (1 - 1 / (i + 2)) * A.gradient(x)

    # Set the performance metric to distance between xN and AxN
    problem.set_performance_metric((x - A.gradient(x)) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (2 / (n + 1)) ** 2

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of Halpern Iterations ***')
        print('\tPEPit guarantee:\t ||xN - AxN||^2 <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||xN - AxN||^2 <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_halpern_iteration(n=25, wrapper="cvxpy", solver=None, verbose=1)
