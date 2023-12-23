import numpy as np

from PEPit import PEP
from PEPit.functions import SmoothConvexFunction


def wc_randomized_coordinate_descent_smooth_convex(L, gamma, d, t, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code computes a worst-case guarantee for **randomized block-coordinate descent** with :math:`d` blocks and
    fixed step-size :math:`\\gamma`.
    That is, it verifies that the Lyapunov function

    .. math:: \\phi(t, x_t) = (t \\gamma \\frac{L}{d} + 1)(f(x_t) - f_\\star) + \\frac{L}{2} \\|x_t - x_\\star\||^2

    is decreasing in expectation over the **randomized block-coordinate descent** algorithm. We use the notation 
    :math:`\\mathbb{E}` for denoting the expectation over the uniform distribution
    of the index :math:`i \\sim \\mathcal{U}\\left([|1, n|]\\right)`.

    In short, for given values of :math:`L`, :math:`d`, and :math:`\\gamma`, it computes the worst-case value
    of :math:`\\mathbb{E}[\\phi(t, x_t)]` such that :math:`\\phi(x_{t-1}) \\leqslant 1`.

    **Algorithm**:
    Randomized block-coordinate descent is described by

    .. math::
        \\begin{eqnarray}
            \\text{Pick random }i & \\sim & \\mathcal{U}\\left([|1, d|]\\right), \\\\
            x_{t+1} & = & x_t - \\gamma \\nabla_i f(x_t),
        \\end{eqnarray}

    where :math:`\\gamma` is a step-size and :math:`\\nabla_i f(x_t)` is the :math:`i^{\\text{th}}` partial gradient.

    **Theoretical guarantee**:
    When :math:`\\gamma \\leqslant \\frac{1}{L}`,
    the **tight** theoretical guarantee can be found in [1, Appendix I, Theorem 16]:

    .. math:: \\mathbb{E}[\\phi(t, x_t)] \\leqslant \\phi(t-1, x_{t-1}),

    where :math:`\\phi(t, x_t) = (t \\gamma \\frac{L}{d} + 1)(f(x_t) - f_\\star) + \\frac{L}{2} \\|x_t - x_\\star\\|^2`.

    **References**:

    `[1] A. Taylor, F. Bach (2019). Stochastic first-order methods: non-asymptotic and computer-aided
    analyses via potential functions. In Conference on Learning Theory (COLT).
    <https://arxiv.org/pdf/1902.00947.pdf>`_

    Args:
        L (float): the smoothness parameter.
        gamma (float): the step-size.
        d (int): the dimension.
        t (int): number of iterations.
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
        >>> L = 1
        >>> pepit_tau, theoretical_tau = wc_randomized_coordinate_descent_smooth_convex(L=L, gamma=1 / L, d=2, n=4, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 6x6
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 12 scalar constraint(s) ...
        			Function 1 : 12 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Setting up the problem: 1 partition(s) added
        			Partition 1 with 2 blocks: Adding 1 scalar constraint(s)...
        			Partition 1 with 2 blocks: 1 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 1.0000000021855517
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 4.888278544731664e-09
        		All the primal scalar constraints are verified up to an error of 8.385744333248845e-09
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 6.347767642940552e-08
        (PEPit) Final upper bound (dual): 1.0000000024690172 and lower bound (primal example): 1.0000000021855517 
        (PEPit) Duality gap: absolute: 2.8346547331636884e-10 and relative: 2.834654726968404e-10
        *** Example file: worst-case performance of randomized  coordinate gradient descent ***
        	PEPit guarantee:	 E[phi(t, x_t)] <= 1.0 phi(t-1, x_(t-1))
        	Theoretical guarantee:	 E[phi(t, x_t)] <= 1.0 phi(t-1, x_(t-1))
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a partition of the ambient space in d blocks of variables
    partition = problem.declare_block_partition(d=d)

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the point x_{t-1} of the algorithm
    xt_minus_1 = problem.set_initial_point()

    # Define the Lyapunov function
    def phi(k, x):
        dk = k * gamma * L / d + 1
        return dk * (func(x) - fs) + L / 2 * (x - xs) ** 2

    # Set the initial condition
    problem.set_initial_condition(phi(t - 1, xt_minus_1) <= 1)

    # Compute all the possible outcomes of the randomized coordinate descent step
    gt_minus_1 = func.gradient(xt_minus_1)
    xt_list = [xt_minus_1 - gamma * partition.get_block(gt_minus_1, i) for i in range(d)]

    # Compute the expected value of the Lyapunov from the different possible outcomes
    phi_t = np.mean([phi(t, xt) for xt in xt_list])

    # Set the performance metric to the variance
    problem.set_performance_metric(phi_t)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 1.

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of randomized  coordinate gradient descent ***')
        print('\tPEPit guarantee:\t E[phi(t, x_t)] <= {:.6} phi(t-1, x_(t-1))'.format(pepit_tau))
        print('\tTheoretical guarantee:\t E[phi(t, x_t)] <= {:.6} phi(t-1, x_(t-1))'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    pepit_tau, theoretical_tau = wc_randomized_coordinate_descent_smooth_convex(L=L, gamma=1 / L, d=2, t=4,
                                                                                wrapper="cvxpy", solver=None,
                                                                                verbose=1)
