from PEPit import PEP
from PEPit.functions import BlockSmoothConvexFunction


def wc_cyclic_coordinate_descent(L, n, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth by blocks (with :math:`d` blocks) and convex.

    This code computes a worst-case guarantee for **cyclic coordinate descent** with fixed step-sizes :math:`1/L_i`.
    That is, it computes the smallest possible :math:`\\tau(n, d, L)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, d, L) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of cyclic coordinate descent with fixed step-sizes :math:`1/L_i`, and
    where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n`, :math:`L`, and :math:`d`, :math:`\\tau(n, d, L)` is computed as
    the worst-case value of :math:`f(x_n)-f_\\star` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    Cyclic coordinate descent is described by

    .. math:: x_{t+1} = x_t - \\frac{1}{L_{i_t}} \\nabla_{i_t} f(x_t),

    where :math:`L_{i_t}` is the Lipschitz constant of the block :math:`i_t`,
    and where :math:`i_t` follows a prescribed ordering.

    **References**:
    
    `[1] Z. Shi, R. Liu (2016).
    Better worst-case complexity analysis of the block coordinate descent method for large scale machine learning.
    In 2017 16th IEEE International Conference on Machine Learning and Applications (ICMLA).
    <https://arxiv.org/pdf/1608.04826.pdf>`_

    Args:
        L (list): list of floats, smoothness parameters (for each block).
        n (int): number of iterations.
        verbose (int): Level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): None

    Example:
        >>> L = [1., 2., 10.]
        >>> pepit_tau, theoretical_tau = wc_cyclic_coordinate_descent(L=L, n=9, verbose=1)
        (PEPit) Setting up the problem: size of the main PSD matrix: 34x34
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
		         function 1 : Adding 330 scalar constraint(s) ...
		         function 1 : 330 scalar constraint(s) added
        (PEPit) Setting up the problem: constraints for 0 function(s)
        (PEPit) Setting up the problem: 1 partition(s) added
		         partition 1 with 3 blocks: Adding 363 scalar constraint(s)...
		         partition 1 with 3 blocks: 363 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); optimal value: 0.6485910874364834
        *** Example file: worst-case performance of cyclic coordinate descent with fixed step-sizes ***
	        PEPit guarantee:	  f(x_n)-f_* <= 0.648591 ||x_0 - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a partition of the ambient space in d blocks of variables
    d = len(L)
    partition = problem.declare_block_partition(d=d)

    # Declare a strongly convex smooth function
    func = problem.declare_function(BlockSmoothConvexFunction, L=L, partition=partition)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the GD method
    x = x0
    for k in range(n):
        i = k % d
        x = x - 1 / L[i] * partition.get_block(func.gradient(x), i)

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric(func(x) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = None

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of cyclic coordinate descent with fixed step-sizes ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = [1., 2., 10.]
    pepit_tau, theoretical_tau = wc_cyclic_coordinate_descent(L=L, n=9, verbose=1)
