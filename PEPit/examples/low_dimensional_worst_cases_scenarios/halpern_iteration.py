from PEPit import PEP
from PEPit.operators import LipschitzOperator


def wc_halpern_iteration(n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the fixed point problem

    .. math:: \\mathrm{Find}\\, x:\\, x = Ax,

    where :math:`A` is a non-expansive operator,
    that is a :math:`L`-Lipschitz operator with :math:`L=1`.

    This code computes a worst-case guarantee for the **Halpern Iteration**, and looks for a low-dimensional
    worst-case example nearly achieving this worst-case guarantee.
    That is, it computes the smallest possible :math:`\\tau(n)` such that the guarantee

    .. math:: \\|x_n - Ax_n\\|^2 \\leqslant \\tau(n) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the **Halpern iteration**,
    and :math:`x_\\star` the fixed point of :math:`A`.

    In short, for a given value of :math:`n`,
    :math:`\\tau(n)` is computed as the worst-case value of
    :math:`\\|x_n - Ax_n\\|^2` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`. Then, it looks for a low-dimensional
    nearly achieving this performance.
    
    **Algorithm**: The Halpern iteration can be written as

        .. math:: x_{t+1} = \\frac{1}{t + 2} x_0 + \\left(1 - \\frac{1}{t + 2}\\right) Ax_t.

    **Theoretical guarantee**: A **tight** worst-case guarantee for Halpern iteration can be found in [1, Theorem 2.1]:

        .. math:: \\|x_n - Ax_n\\|^2 \\leqslant \\left(\\frac{2}{n+1}\\right)^2 \\|x_0 - x_\\star\\|^2.

    **References**: The detailed approach and tight bound are available in [1].

    `[1] F. Lieder (2021). On the convergence rate of the Halpern-iteration. Optimization Letters, 15(2), 405-418.
    <http://www.optimization-online.org/DB_FILE/2017/11/6336.pdf>`_

    `[2] F. Maryam, H. Hindi, S. Boyd (2003). Log-det heuristic for matrix rank minimization with applications to Hankel
    and Euclidean distance matrices. American Control Conference (ACC).
    <https://web.stanford.edu/~boyd/papers/pdf/rank_min_heur_hankel.pdf>`_

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
        >>> pepit_tau, theoretical_tau = wc_halpern_iteration(n=10, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 13x13
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 66 scalar constraint(s) ...
        			Function 1 : 66 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.03305790577349196
        (PEPit) Postprocessing: 12 eigenvalue(s) > 0 before dimension reduction
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.033047905727429217
        (PEPit) Postprocessing: 1 eigenvalue(s) > 3.494895432655832e-09 after 1 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.03304790606633461
        (PEPit) Postprocessing: 1 eigenvalue(s) > 0 after 2 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.03304790577532096
        (PEPit) Postprocessing: 1 eigenvalue(s) > 1.834729915591876e-13 after 3 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.03304790577532096
        (PEPit) Postprocessing: 1 eigenvalue(s) > 1.834729915591876e-13 after dimension reduction
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 1.3350847983797365e-12
        		All the primal scalar constraints are verified up to an error of 5.296318938974309e-13
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 5.146036222356589e-09
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 1.81631899443615e-07
        (PEPit) Final upper bound (dual): 0.03305791391531532 and lower bound (primal example): 0.03304790577532096 
        (PEPit) Duality gap: absolute: 1.0008139994355236e-05 and relative: 0.00030283734353385173
        *** Example file: worst-case performance of Halpern Iterations ***
        	PEPit guarantee:	 ||xN - AxN||^2 == 0.0330579 ||x0 - x_*||^2
        	Theoretical guarantee:	 ||xN - AxN||^2 <= 0.0330579 ||x0 - x_*||^2
    
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
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose,
                              dimension_reduction_heuristic="logdet3",
                              tol_dimension_reduction=1e-5)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (2 / (n + 1)) ** 2

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of Halpern Iterations ***')
        print('\tPEPit guarantee:\t ||xN - AxN||^2 == {:.6} ||x0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||xN - AxN||^2 <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_halpern_iteration(n=10, wrapper="cvxpy", solver=None, verbose=1)
