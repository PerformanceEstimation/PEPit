import numpy as np

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def wc_randomized_coordinate_descent_smooth_strongly_convex(L, mu, gamma, d, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for **randomized block-coordinate descent**
    with step-size :math:`\\gamma`.
    That is, it computes the smallest possible :math:`\\tau(L, \\mu, \\gamma, d)` such that the guarantee

    .. math:: \\mathbb{E}[\\|x_{t+1} - x_\star \\|^2] \\leqslant \\tau(L, \\mu, \\gamma, d) \\|x_t - x_\\star\\|^2
    
    holds for any fixed step-size :math:`\\gamma` and any number of blocks :math:`d`,
    and where :math:`x_\\star` denotes a minimizer of :math:`f`. The notation :math:`\\mathbb{E}`
    denotes the expectation over the uniform distribution of the index
    :math:`i \\sim \\mathcal{U}\\left([|1, n|]\\right)`.

    In short, for given values of :math:`\\mu`, :math:`L`, :math:`d`, and :math:`\\gamma`,
    :math:`\\tau(L, \\mu, \\gamma, d)` is computed as the worst-case value of
    :math:`\\mathbb{E}[\\|x_{t+1} - x_\star \\|^2]` when :math:`\\|x_t - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    Randomized block-coordinate descent is described by

    .. math::
        \\begin{eqnarray}
            \\text{Pick random }i & \\sim & \\mathcal{U}\\left([|1, d|]\\right), \\\\
            x_{t+1} & = & x_t - \\gamma \\nabla_i f(x_t),
        \\end{eqnarray}

    where :math:`\\gamma` is a step-size and :math:`\\nabla_i f(x_t)` is the :math:`i^{\\text{th}}` partial gradient.

    **Theoretical guarantee**:
    When :math:`\\gamma \\leqslant \\frac{1}{L}`, the **tight** theoretical guarantee
    can be found in [1, Appendix I, Theorem 17]:

    .. math:: \\mathbb{E}[\\|x_{t+1} - x_\star \\|^2] \\leqslant \\rho^2 \\|x_t-x_\\star\\|^2,

    where :math:`\\rho^2 = \\max \\left( \\frac{(\\gamma\\mu - 1)^2 + d - 1}{d},\\frac{(\\gamma L - 1)^2 + d - 1}{d} \\right)`.

    **References**:

    `[1] A. Taylor, F. Bach (2019). Stochastic first-order methods: non-asymptotic and computer-aided
    analyses via potential functions. In Conference on Learning Theory (COLT).
    <https://arxiv.org/pdf/1902.00947.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong-convexity parameter.
        gamma (float): the step-size.
        d (int): the dimension.
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
        >>> mu = 0.1
        >>> gamma = 2 / (mu + L)
        >>> pepit_tau, theoretical_tau = wc_randomized_coordinate_descent_smooth_strongly_convex(L=L, mu=mu, gamma=gamma, d=2, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 4x4
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 2 scalar constraint(s) ...
        			Function 1 : 2 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Setting up the problem: 1 partition(s) added
        			Partition 1 with 2 blocks: Adding 1 scalar constraint(s)...
        			Partition 1 with 2 blocks: 1 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.8347107438584297
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite
        		All the primal scalar constraints are verified up to an error of 1.4183154650737606e-11
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 4.950747594358786e-10
        (PEPit) Final upper bound (dual): 0.8347107438665666 and lower bound (primal example): 0.8347107438584297 
        (PEPit) Duality gap: absolute: 8.136935569780235e-12 and relative: 9.748209939370677e-12
        *** Example file: worst-case performance of randomized coordinate gradient descent ***
        	PEPit guarantee:	 E[||x_(t+1) - x_*||^2] <= 0.834711 ||x_t - x_*||^2
        	Theoretical guarantee:	 E[||x_(t+1) - x_*||^2] <= 0.834711 ||x_t - x_*||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a partition of the ambient space in d blocks of variables
    partition = problem.declare_block_partition(d=d)

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute all the possible outcomes of the randomized coordinate descent step
    g0 = func.gradient(x0)
    x1_list = [x0 - gamma * partition.get_block(g0, i) for i in range(d)]

    # Set the performance metric to the expected value of the distance to optimiser from the different possible outcomes
    problem.set_performance_metric(np.mean([(x1 - xs) ** 2 for x1 in x1_list]))

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = max(((mu * gamma - 1) ** 2 + d - 1) / d, ((L * gamma - 1) ** 2 + d - 1) / d)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of randomized coordinate gradient descent ***')
        print('\tPEPit guarantee:\t E[||x_(t+1) - x_*||^2] <= {:.6} ||x_t - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t E[||x_(t+1) - x_*||^2] <= {:.6} ||x_t - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    mu = 0.1
    gamma = 2 / (mu + L)
    pepit_tau, theoretical_tau = wc_randomized_coordinate_descent_smooth_strongly_convex(L=L, mu=mu, gamma=gamma, d=2,
                                                                                         wrapper="cvxpy", solver=None,
                                                                                         verbose=1)
