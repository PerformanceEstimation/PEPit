from PEPit import PEP
from PEPit.functions import BlockSmoothConvexFunction


def wc_gradient_descent_blocks(L, n, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth by blocks (with :math:`d` blocks) and convex.

    This code computes a worst-case guarantee for **gradient descent** with fixed step-size :math:`1/\\max_i L_i`.
    That is, it computes the smallest possible :math:`\\tau(n, L)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of gradient descent with fixed step-size, and
    where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n`, and :math:`L`, :math:`\\tau(n, L)` is computed as the worst-case
    value of :math:`f(x_n)-f_\\star` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{t+1} = x_t - \\frac{1}{\\max_i L_i} \\nabla f(x_t),

    where :math:`L_i`'s are the Lipschitz constants.

    **Theoretical guarantee**:
    A **tight** theoretical guarantee can be found in [1, Theorem 3.1]:

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{\\max_i L_i}{4n+2} \\|x_0-x_\\star\\|^2.

    **References**:

    `[1] Y. Drori, M. Teboulle (2014). Performance of first-order methods for smooth convex minimization: a novel
    approach. Mathematical Programming 145(1–2), 451–482.
    <https://arxiv.org/pdf/1206.3209.pdf>`_

    Args:
        L (list): list of smoothness parameters.
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
        >>> L = [1., 2., 10.]
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_blocks(L=L, n=3, verbose=1)
        (PEPit) Setting up the problem: size of the main PSD matrix: 16x16
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
		         function 1 : Adding 60 scalar constraint(s) ...
		         function 1 : 60 scalar constraint(s) added
        (PEPit) Setting up the problem: constraints for 0 function(s)
        (PEPit) Setting up the problem: 1 partition(s) added
		         partition 1 with 3 blocks: Adding 75 scalar constraint(s)...
		         partition 1 with 3 blocks: 75 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); optimal value: 0.6632652826522251
        *** Example file: worst-case performance of gradient descent (block smoothness) ***
	        PEPit guarantee:          f(x_n)-f_* <= 0.663265 ||x_0 - x_*||^2
	        Theoretical guarantee:    f(x_n)-f_* <= 0.714286 ||x_0 - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()
    
    # Declare a partition of the ambient space in d blocks of variables
    d = len(L)
    partition = problem.declare_block_partition(d=d)
    Lmax = max(L)

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
    gamma = 1/Lmax
    for _ in range(n):
        x = x - gamma * func.gradient(x)

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric(func(x) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = Lmax / (2 * (2 * n + 1))

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent (block smoothness) ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = [1., 2., 10.]
    pepit_tau, theoretical_tau = wc_gradient_descent_blocks(L=L, n=3, verbose=1)
