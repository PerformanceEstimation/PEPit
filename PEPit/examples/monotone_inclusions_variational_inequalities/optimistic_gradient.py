from PEPit import PEP
from PEPit.functions import ConvexIndicatorFunction
from PEPit.operators import LipschitzStronglyMonotoneOperator
from PEPit.primitive_steps import proximal_step


def wc_optimistic_gradient(n, gamma, L, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the monotone variational inequality

        .. math:: \\mathrm{Find}\\, x_\\star \\in C\\text{ such that } \\left<F(x_\\star);x-x_\\star\\right> \\geqslant 0\\,\\,\\forall x\\in C,

    where :math:`C` is a closed convex set and :math:`F` is maximally monotone and Lipschitz.

    This code computes a worst-case guarantee for the **optimistic gradient method**.
    That, it computes the smallest possible :math:`\\tau(n)` such that the guarantee

        .. math:: \\|\\tilde{x}_n - \\tilde{x}_{n-1}\\|^2 \\leqslant \\tau(n) \\|x_0 - x_\\star\\|^2,

    is valid, where :math:`\\tilde{x}_n` is the output of the **optimistic gradient method**
    and :math:`x_0` its starting point.

    **Algorithm**: The optimistic gradient method is described as follows, for :math:`t \in \\{ 0, \\dots, n-1\\}`,
    
    .. math::
            :nowrap:

            \\begin{eqnarray}
                 \\tilde{x}_{t} & = & \\mathrm{Proj}_{C} [x_t-\\gamma F(\\tilde{x}_{t-1})], \\\\
                 {x}_{t+1} & = & \\tilde{x}_t + \\gamma (F(\\tilde{x}_{t-1}) - F(\\tilde{x}_t)).
            \\end{eqnarray}

    where :math:`\\gamma` is some step-size.

    **Theoretical guarantee**: The method and many variants of it are discussed in [1] and a PEP formulation suggesting
    a worst-case guarantee in :math:`O(1/n)` can be found in [2, Appendix D].
    
    **References**:
    
    `[1] Y.-G. Hsieh, F. Iutzeler, J. Malick, P. Mertikopoulos (2019).
    On the convergence of single-call stochastic extra-gradient methods.
    Advances in Neural Information Processing Systems, 32:6938–6948, 2019
    <https://arxiv.org/pdf/1908.08465.pdf>`_

    `[2] E. Gorbunov, A. Taylor, G. Gidel (2022).
    Last-Iterate Convergence of Optimistic Gradient Method for Monotone Variational Inequalities.
    <https://arxiv.org/pdf/2205.08446.pdf>`_

    Args:
        n (int): number of iterations.
        gamma (float): the step-size.
        L (float): the Lipschitz constant.
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + solver details.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (None): no theoretical bound.

    Example:
        >>> pepit_tau, theoretical_tau = wc_optimistic_gradient(n=5, gamma=1 / 4, L=1, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 15x15
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
        			Function 1 : Adding 49 scalar constraint(s) ...
        			Function 1 : 49 scalar constraint(s) added
        			Function 2 : Adding 42 scalar constraint(s) ...
        			Function 2 : 42 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.06631412698565144
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite
        		All the primal scalar constraints are verified up to an error of 2.881525884221303e-09
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 2.0104852598258245e-08
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 9.484355451749491e-08
        (PEPit) Final upper bound (dual): 0.06631413197861648 and lower bound (primal example): 0.06631412698565144 
        (PEPit) Duality gap: absolute: 4.992965041417108e-09 and relative: 7.529263021885893e-08
        *** Example file: worst-case performance of the Optimistic Gradient Method***
        	PEPit guarantee:	 ||x(n) - x(n-1)||^2 <= 0.0663141 ||x0 - xs||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare an indicator function and a monotone operator
    ind_C = problem.declare_function(ConvexIndicatorFunction)
    F = problem.declare_function(LipschitzStronglyMonotoneOperator, mu=0, L=L)

    total_problem = F + ind_C

    # Start by defining its unique optimal point xs = x_*
    xs = total_problem.stationary_point()

    # Then define the starting point x0 of the algorithm and its gradient value g0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute n steps of the Proximal Gradient method starting from x0
    x, _, _ = proximal_step(x0, ind_C, gamma)
    xtilde = x
    V = F.gradient(xtilde)
    for _ in range(n):
        previous_xtilde = xtilde
        xtilde, _, _ = proximal_step(x - gamma * V, ind_C, gamma)
        previous_V = V
        V = F.gradient(xtilde)
        x = xtilde + gamma * (previous_V - V)

    # Set the performance metric to the distance between x(n) and x(n-1)
    problem.set_performance_metric((xtilde - previous_xtilde) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = None

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the Optimistic Gradient Method***')
        print('\tPEPit guarantee:\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_optimistic_gradient(n=5, gamma=1 / 4, L=1, wrapper="cvxpy", solver=None, verbose=1)
