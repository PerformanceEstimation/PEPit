from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def wc_gradient_descent_useless_blocks(L, gamma, n, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code computes a worst-case guarantee for **gradient descent** with fixed step-size :math:`\\gamma`.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\gamma)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L, \\gamma) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of gradient descent with fixed step-size :math:`\\gamma`, and
    where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n`, :math:`L`, and :math:`\\gamma`, :math:`\\tau(n, L, \\gamma)` is computed as the worst-case
    value of :math:`f(x_n)-f_\\star` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma \\nabla f(x_t),

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**:
    When :math:`\\gamma \\leqslant \\frac{1}{L}`, the **tight** theoretical guarantee can be found in [1, Theorem 3.1]:

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{L}{4nL\\gamma+2} \\|x_0-x_\\star\\|^2,

    which is tight on some Huber loss functions.

    **References**:

    `[1] Y. Drori, M. Teboulle (2014). Performance of first-order methods for smooth convex minimization: a novel
    approach. Mathematical Programming 145(1–2), 451–482.
    <https://arxiv.org/pdf/1206.3209.pdf>`_

    Args:
        L (float): the smoothness parameter.
        gamma (float): step-size.
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
        >>> L = 1
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_useless_blocks(L=L, gamma=1 / L, n=5, verbose=1)
        (PEPit) Setting up the problem: size of the main PSD matrix: 8x8
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
		         function 1 : Adding 42 scalar constraint(s) ...
		         function 1 : 42 scalar constraint(s) added
        (PEPit) Setting up the problem: constraints for 0 function(s)
        (PEPit) Setting up the problem: 1 partition(s) added
                        partition 1 with 1 blocks: Adding 0 scalar constraint(s)...
                        partition 1 with 1 blocks: 0 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); optimal value: 0.04545454238767727
        *** Example file: worst-case performance of gradient descent with fixed step-sizes ***
	        PEPit guarantee:	   f(x_n)-f_* <= 0.0454545 ||x_0 - x_*||^2
	        Theoretical guarantee:    f(x_n)-f_* <= 0.0454545 ||x_0 - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()
    
    # Declare a partition of the ambient space in d blocks of variables
    partition = problem.declare_block_partition(d=1)

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=0, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = partition.get_block(func.stationary_point(),0)
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = partition.get_block(problem.set_initial_point(),0)

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the GD method
    x = x0
    for _ in range(n):
        x = x - gamma * partition.get_block(func.gradient(x),0)

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric(func(x) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / (2 * (2 * n * L * gamma + 1))

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent with fixed step-sizes ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    pepit_tau, theoretical_tau = wc_gradient_descent_useless_blocks(L=L, gamma=1 / L, n=5, verbose=1)
