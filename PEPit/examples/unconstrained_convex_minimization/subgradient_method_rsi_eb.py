from PEPit import PEP
from PEPit.functions import RsiEbFunction


def wc_subgradient_method_rsi_eb(mu, L, gamma, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` verifies the "lower" restricted secant inequality (:math:`\\mu-\\text{RSI}^-`)
    and the "upper" error bound (:math:`L-\\text{EB}^+`) [1].

    This code computes a worst-case guarantee for **gradient descent** with fixed step-size :math:`\\gamma`.
    That is, it computes the smallest possible :math:`\\tau(n, \\mu, L, \\gamma)` such that the guarantee

    .. math:: \\| x_n - x_\\star \\|^2 \\leqslant \\tau(n, \\mu, L, \\gamma) \\| x_0 - x_\\star \\|^2

    is valid, where :math:`x_n` is the output of gradient descent with fixed step-size :math:`\\gamma`, and
    where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n`, :math:`L`, and :math:`\\gamma`,
    :math:`\\tau(n, \\mu, L, \\gamma)` is computed as the worst-case value of
    :math:`\\| x_n - x_\\star \\|^2` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    Sub-gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma \\nabla f(x_t),

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**:
    The **tight** theoretical guarantee can be found in [1, Prop 1] (upper bound) and [1, Theorem 2] (lower bound):

    .. math:: \\| x_n - x_\\star \\|^2 \\leqslant (1 - 2\\gamma\\mu + L^2 \\gamma^2)^n \\|x_0-x_\\star\\|^2.

    **References**:
    Definition and convergence guarantees can be found in [1].

    `[1] C. Guille-Escuret, B. Goujaud, A. Ibrahim, I. Mitliagkas (2022).
    Gradient Descent Is Optimal Under Lower Restricted Secant Inequality And Upper Error Bound.
    <https://arxiv.org/pdf/2203.00342.pdf>`_

    Args:
        mu (float): the rsi parameter
        L (float): the eb parameter.
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
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> mu = .1
        >>> L = 1
        >>> pepit_tau, theoretical_tau = wc_subgradient_method_rsi_eb(mu=mu, L=L, gamma=mu / L ** 2, n=4, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 6x6
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 8 scalar constraint(s) ...
        			Function 1 : 8 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.9605960099986828
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite
        		All the primal scalar constraints are verified
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 2.9007807847382036e-12
        (PEPit) Final upper bound (dual): 0.960596009998838 and lower bound (primal example): 0.9605960099986828 
        (PEPit) Duality gap: absolute: 1.5520917884259688e-13 and relative: 1.6157591456455218e-13
        *** Example file: worst-case performance of gradient descent with fixed step-sizes ***
        	PEPit guarantee:	 f(x_n)-f_* <= 0.960596 ||x_0 - x_*||^2
        	Theoretical guarantee:	 f(x_n)-f_* <= 0.960596 ||x_0 - x_*||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(RsiEbFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the GD method
    x = x0
    for _ in range(n):
        x = x - gamma * func.gradient(x)

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric((x - xs) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (1 - 2 * gamma * mu + gamma ** 2 * L ** 2) ** n

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent with fixed step-sizes ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    mu = .1
    L = 1
    pepit_tau, theoretical_tau = wc_subgradient_method_rsi_eb(mu=mu, L=L, gamma=mu / L ** 2, n=4,
                                                              wrapper="cvxpy", solver=None,
                                                              verbose=1)
