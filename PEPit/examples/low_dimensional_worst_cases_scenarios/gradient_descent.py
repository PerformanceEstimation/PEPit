from PEPit import PEP
from PEPit.functions import SmoothFunction


def wc_gradient_descent(L, gamma, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth.

    This code computes a worst-case guarantee for **gradient descent** with fixed step-size :math:`\\gamma`,
    and looks for a low-dimensional worst-case example nearly achieving this worst-case guarantee.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\gamma)` such that the guarantee

    .. math:: \\min_{t\\leqslant n} \\|\\nabla f(x_t)\\|^2 \\leqslant \\tau(n, L, \\gamma) (f(x_0) - f(x_n))

    is valid, where :math:`x_n` is the n-th iterates obtained with the gradient method with fixed step-size.
    Then, it looks for a low-dimensional nearly achieving this performance.

    **Algorithm**:
    Gradient descent is described as follows, for :math:`t \in \\{ 0, \\dots, n-1\\}`,

    .. math:: x_{t+1} = x_t - \\gamma \\nabla f(x_t),

    where :math:`\\gamma` is a step-size and.

    **Theoretical guarantee**:
    When :math:`\\gamma \\leqslant \\frac{1}{L}`, an empirically tight theoretical worst-case guarantee is

    .. math:: \\min_{t\\leqslant n} \\|\\nabla f(x_t)\\|^2 \\leqslant \\frac{4}{3}\\frac{L}{n} (f(x_0) - f(x_n)),

    see discussions in [1, page 190] and [2].

    **References**:

    `[1] Taylor, A. B. (2017). Convex interpolation and performance estimation of first-order methods for
    convex optimization. PhD Thesis, UCLouvain.
    <https://dial.uclouvain.be/downloader/downloader.php?pid=boreal:182881&datastream=PDF_01>`_

    `[2] H. Abbaszadehpeivasti, E. de Klerk, M. Zamani (2021). The exact worst-case convergence rate of the
    gradient method with fixed step lengths for L-smooth functions. Optimization Letters, 16(6), 1649-1661.
    <https://arxiv.org/pdf/2104.05468v3.pdf>`_

    Args:
        L (float): the smoothness parameter.
        gamma (float): step-size.
        n (int): number of iterations.
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + solver details.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (float): theoretical value.

    Example:
        >>> L = 1
        >>> gamma = 1 / L
        >>> pepit_tau, theoretical_tau = wc_gradient_descent(L=L, gamma=gamma, n=5, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 7x7
        (PEPit) Setting up the problem: performance measure is the minimum of 6 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 30 scalar constraint(s) ...
        			Function 1 : 30 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.2666666655116666
        (PEPit) Postprocessing: 7 eigenvalue(s) > 0 before dimension reduction
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.26656665923389067
        (PEPit) Postprocessing: 1 eigenvalue(s) > 1.1259924068173275e-07 after 1 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.2665666687640418
        (PEPit) Postprocessing: 1 eigenvalue(s) > 0 after 2 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.2665666687640418
        (PEPit) Postprocessing: 1 eigenvalue(s) > 0 after dimension reduction
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 6.171828087683806e-11
        		All the primal scalar constraints are verified
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 4.5045553432130066e-10
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 4.49157760320874e-09
        (PEPit) Final upper bound (dual): 0.2666666657156721 and lower bound (primal example): 0.2665666687640418 
        (PEPit) Duality gap: absolute: 9.999695163032118e-05 and relative: 0.0003751292391279272
        *** Example file: worst-case performance of gradient descent with fixed step-size ***
        	PEPit guarantee:	 min_i ||f'(x_i)||^2 == 0.266667 (f(x_0)-f_*)
        	Theoretical guarantee:	 min_i ||f'(x_i)||^2 <= 0.266667 (f(x_0)-f_*)
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothFunction, L=L)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value g0 and f0
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Run n steps of GD method with step-size gamma
    x = x0
    gx, fx = g0, f0

    # Set the performance metric to the minimum of the gradient norm over the iterations
    problem.set_performance_metric(gx ** 2)

    for i in range(n):
        x = x - gamma * gx
        # Set the performance metric to the minimum of the gradient norm over the iterations
        gx, fx = func.oracle(x)
        problem.set_performance_metric(gx ** 2)

    # Set the initial constraint that is the difference between fN and f0
    problem.set_initial_condition(f0 - fx <= 1)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose,
                              dimension_reduction_heuristic="logdet2")

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 4 / 3 * L / n

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent with fixed step-size ***')
        print('\tPEPit guarantee:\t min_i ||f\'(x_i)||^2 == {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t min_i ||f\'(x_i)||^2 <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    gamma = 1 / L
    pepit_tau, theoretical_tau = wc_gradient_descent(L=L, gamma=gamma, n=5, wrapper="cvxpy", solver=None, verbose=1)
