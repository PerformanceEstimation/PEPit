from PEPit import PEP
from PEPit.operators import StronglyMonotoneOperator
from PEPit.primitive_steps import proximal_step


def phi(mu, idx):
    if idx == -1:
        return 0
    return ((1 + 2 * mu) ** (2 * idx + 2) - 1) / ((1 + 2 * mu) ** 2 - 1)


def wc_optimal_strongly_monotone_proximal_point(n, mu, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the monotone inclusion problem

    .. math:: \\mathrm{Find}\\, x:\\, 0\\in Ax,

    where :math:`A` is maximally :math:`\\mu`-strongly monotone. 
    We denote by :math:`J_{A}` the resolvent of :math:`A`.

    For any :math:`x` such that :math:`x = J_{A} y` for some :math:`y`,
    define the resolvent residual :math:`\\tilde{A}x = y - J_{A}y \\in Ax`.

    This code computes a worst-case guarantee for the **Optimal Strongly-monotone Proximal Point Method** (OS-PPM). 
    That is, it computes the smallest possible :math:`\\tau(n, \\mu)` such that the guarantee 

    .. math:: \\|\\tilde{A}x_n\\|^2 \\leqslant \\tau(n, \\mu) \\|x_0 - x_\\star\\|^2,

    is valid, where :math:`x_n` is the output of the **Optimal Strongly-monotone Proximal Point Method**,
    and :math:`x_\\star` is a zero of :math:`A`. In short, for a given value of :math:`n, \\mu`,
    :math:`\\tau(n, \\mu)` is computed as the worst-case value of
    :math:`\\|\\tilde{A}x_n\\|^2` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**: The Optimal Strongly-monotone Proximal Point Method can be written as

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_{t+1} & = & J_{A} y_t,\\\\
                y_{t+1} & = & x_{t+1} + \\frac{\\varphi_{t} - 1}{\\varphi_{t+1}} (x_{t+1} - x_t) - \\frac{2 \\mu \\varphi_{t}}{\\varphi_{t+1}} (y_t - x_{t+1}) \\\\
                         &  & + \\frac{(1+2\\mu) \\varphi_{t-1}}{\\varphi_{t+1}} (y_{t-1} - x_t). 
            \\end{eqnarray}

    where :math:`\\varphi_k = \sum_{i=0}^k (1+2\\mu)^{2i}` with :math:`\\varphi_{-1}=0`
    and :math:`x_0 = y_0 = y_{-1}` is a starting point.

    This method is equivalent to the Optimal Contractive Halpern iteration.

    **Theoretical guarantee**: A **tight** worst-case guarantee for the Optimal Strongly-monotone Proximal Point Method
    can be found in [1, Theorem 3.2, Corollary 4.2]:

        .. math:: \\|\\tilde{A}x_n\\|^2 \\leqslant \\left( \\frac{1}{\sum_{k=0}^{N-1} (1+2\\mu)^k} \\right)^2 \\|x_0 - x_\\star\\|^2.

    **References**: The detailed approach and tight bound are available in [1].

    `[1] J. Park, E. Ryu (2022).
    Exact Optimal Accelerated Complexity for Fixed-Point Iterations.
    In 39th International Conference on Machine Learning (ICML).
    <https://proceedings.mlr.press/v162/park22c/park22c.pdf>`_

    Args:
        n (int): number of iterations.
        mu (float): :math:`\\mu \ge 0`. :math:`A` will be maximal :math:`\\mu`-strongly monotone.
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
        >>> pepit_tau, theoretical_tau = wc_optimal_strongly_monotone_proximal_point(n=10, mu=0.05, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 12x12
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 55 scalar constraint(s) ...
        			Function 1 : 55 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.003936989547244047
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 1.0607807556608727e-09
        		All the primal scalar constraints are verified up to an error of 3.5351675688243754e-09
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 1.804099035739487e-08
        (PEPit) Final upper bound (dual): 0.003936990621254958 and lower bound (primal example): 0.003936989547244047 
        (PEPit) Duality gap: absolute: 1.0740109105886186e-09 and relative: 2.7280004117370406e-07
        *** Example file: worst-case performance of Optimal Strongly-monotone Proximal Point Method ***
        	PEPit guarantee:	 ||AxN||^2 <= 0.00393699 ||x0 - x_*||^2
        	Theoretical guarantee:	 ||AxN||^2 <= 0.00393698 ||x0 - x_*||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a monotone operator
    A = problem.declare_function(StronglyMonotoneOperator, mu=mu)

    # Start by defining the zero point xs
    xs = A.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the difference between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of Optimal Strongly-monotone Proximal Point Method
    x, y, y_prv = x0, x0, x0
    for i in range(n):
        x_nxt, _, _ = proximal_step(y, A, 1)
        y_nxt = x_nxt + (phi(mu, i) - 1) / phi(mu, i + 1) * (x_nxt - x) - 2 * mu * phi(mu, i) / phi(mu, i + 1) * (
                y - x_nxt) + (1 + 2 * mu) * phi(mu, i - 1) / phi(mu, i + 1) * (y_prv - x)
        x, y_prv, y = x_nxt, y, y_nxt

    # Set the performance metric to length of \tilde{A}xN
    problem.set_performance_metric((y_prv - x) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (2 * mu / ((1 + 2 * mu) ** n - 1)) ** 2

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of Optimal Strongly-monotone Proximal Point Method ***')
        print('\tPEPit guarantee:\t ||AxN||^2 <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||AxN||^2 <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_optimal_strongly_monotone_proximal_point(n=10, mu=0.05,
                                                                             wrapper="cvxpy", solver=None,
                                                                             verbose=1)
