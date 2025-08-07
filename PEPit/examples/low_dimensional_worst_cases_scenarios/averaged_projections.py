from PEPit import PEP
from PEPit.functions import ConvexIndicatorFunction
from PEPit.primitive_steps import proximal_step


def wc_averaged_projections(n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex feasibility problem:

    .. math:: \\mathrm{Find}\\, x\\in Q_1\\cap Q_2

    where :math:`Q_1` and :math:`Q_2` are two closed convex sets.

    This code computes a worst-case guarantee for the **averaged projection method**, and looks for a low-dimensional
    worst-case example nearly achieving this worst-case guarantee.
    That is, it computes the smallest possible :math:`\\tau(n)` such that the guarantee

    .. math:: \\|\\mathrm{Proj}_{Q_1}(x_n)-\\mathrm{Proj}_{Q_2}(x_n)\\|^2 \\leqslant \\tau(n) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the **averaged projection method**,
    and :math:`x_\\star\\in Q_1\\cap Q_2` is a solution to the convex feasibility problem.

    In short, for a given value of :math:`n`,
    :math:`\\tau(n)` is computed as the worst-case value of
    :math:`\\|\\mathrm{Proj}_{Q_1}(x_n)-\\mathrm{Proj}_{Q_2}(x_n)\\|^2`
    when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.
    Then, it looks for a low-dimensional nearly achieving this performance.
    
    **Algorithm**: The averaged projection method can be written as

        .. math::
            \\begin{eqnarray}
                x_{t+1} & = & \\frac{1}{2} \\left(\\mathrm{Proj}_{Q_1}(x_t) + \\mathrm{Proj}_{Q_2}(x_t)\\right).
            \\end{eqnarray}

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
        theoretical_tau (None): no theoretical value.

    Example:
        >>> pepit_tau, theoretical_tau = wc_averaged_projections(n=10, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 25x25
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
        			Function 1 : Adding 144 scalar constraint(s) ...
        			Function 1 : 144 scalar constraint(s) added
        			Function 2 : Adding 144 scalar constraint(s) ...
        			Function 2 : 144 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.0684488573474302
        (PEPit) Postprocessing: 4 eigenvalue(s) > 7.89314275744705e-10 before dimension reduction
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.06834885794771604
        (PEPit) Postprocessing: 2 eigenvalue(s) > 6.675793636702899e-09 after 1 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.06834885794771604
        (PEPit) Postprocessing: 2 eigenvalue(s) > 6.675793636702899e-09 after dimension reduction
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 3.4702057470351053e-10
        		All the primal scalar constraints are verified up to an error of 2.313877735249381e-10
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 9.760213634525396e-13
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 3.0486379622054575e-10
        (PEPit) Final upper bound (dual): 0.06844885735562138 and lower bound (primal example): 0.06834885794771604 
        (PEPit) Duality gap: absolute: 9.99994079053379e-05 and relative: 0.001463073574423631
        *** Example file: worst-case performance of the averaged projection method ***
        	PEPit guarantee:	 ||Proj_Q1 (xn) - Proj_Q2 (xn)||^2 == 0.0684489 ||x0 - x_*||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare the two indicator functions and the feasibility problem
    ind_Q1 = problem.declare_function(ConvexIndicatorFunction)
    ind_Q2 = problem.declare_function(ConvexIndicatorFunction)
    func = ind_Q1 + ind_Q2

    # Start by defining a solution xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Run the averaged projection method
    x = x0
    for _ in range(n):
        y1, _, _ = proximal_step(x, ind_Q1, 1)
        y2, _, _ = proximal_step(x, ind_Q2, 1)
        x = 1 / 2 * (y1 + y2)

    # Set the performance metric
    proj1_x, _, _ = proximal_step(x, ind_Q1, 1)
    proj2_x, _, _ = proximal_step(x, ind_Q2, 1)
    problem.set_performance_metric((proj2_x - proj1_x) ** 2)
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose,
                              dimension_reduction_heuristic="logdet1")
    theoretical_tau = None

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the averaged projection method ***')
        print('\tPEPit guarantee:\t ||Proj_Q1 (xn) - Proj_Q2 (xn)||^2 == {:.6} ||x0 - x_*||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_averaged_projections(n=10, wrapper="cvxpy", solver=None, verbose=1)
