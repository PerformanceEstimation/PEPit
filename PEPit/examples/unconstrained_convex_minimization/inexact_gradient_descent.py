from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.primitive_steps import inexact_gradient_step


def wc_inexact_gradient_descent(L, mu, epsilon, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for the **inexact gradient** method.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu, \\varepsilon)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L, \\mu, \\varepsilon) (f(x_0) - f_\\star)

    is valid, where :math:`x_n` is the output of the **inexact gradient** method,
    and where :math:`x_\\star` is the minimizer of :math:`f`.
    In short, for given values of :math:`n`, :math:`L`, :math:`\\mu` and :math:`\\varepsilon`,
    :math:`\\tau(n, L, \\mu, \\varepsilon)` is computed as the worst-case value of
    :math:`f(x_n)-f_\\star` when :math:`f(x_0) - f_\\star \\leqslant 1`.

    **Algorithm**:

        .. math:: x_{t+1} = x_t - \\gamma d_t

        with

        .. math:: \|d_t - \\nabla f(x_t)\| \\leqslant  \\varepsilon \|\\nabla f(x_t)\|

        and

        .. math:: \\gamma = \\frac{2}{L_{\\varepsilon} + \\mu_{\\varepsilon}}

        where :math:`L_{\\varepsilon} = (1 + \\varepsilon) L` and :math:`\\mu_{\\varepsilon} = (1 - \\varepsilon) \\mu`.

    **Theoretical guarantee**:

    The **tight** worst-case guarantee obtained in [1, Theorem 5.3] or [2, Remark 1.6] is

        .. math:: f(x_n) - f_\\star \\leqslant \\left(\\frac{L_{\\varepsilon}-\\mu_{\\varepsilon}}{L_{\\varepsilon}+\\mu_{\\varepsilon}}\\right)^{2n}(f(x_0) - f_\\star),

    where tightness is achieved on simple quadratic functions.

    **References**: The detailed analyses can be found in [1, 2].

    `[1] E. De Klerk, F. Glineur, A. Taylor (2020).
    Worst-case convergence analysis of inexact gradient
    and Newton methods through semidefinite programming performance estimation.
    SIAM Journal on Optimization, 30(3), 2053-2082.
    <https://arxiv.org/pdf/1709.05191.pdf>`_

    `[2] O. Gannot (2021).
    A frequency-domain analysis of inexact gradient methods.
    Mathematical Programming.
    <https://arxiv.org/pdf/1912.13494.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        epsilon (float): level of inaccuracy.
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
        >>> pepit_tau, theoretical_tau = wc_inexact_gradient_descent(L=1, mu=.1, epsilon=.1, n=2, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 7x7
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 12 scalar constraint(s) ...
        			Function 1 : 12 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 1 function(s)
        			Function 1 : Adding 2 scalar constraint(s) ...
        			Function 1 : 2 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.5189167048760179
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 3.328901122905122e-09
        		All the primal scalar constraints are verified up to an error of 9.223752428511034e-09
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 1.0409575365469605e-07
        (PEPit) Final upper bound (dual): 0.5189166992915334 and lower bound (primal example): 0.5189167048760179 
        (PEPit) Duality gap: absolute: -5.584484541465429e-09 and relative: -1.0761813001953176e-08
        *** Example file: worst-case performance of inexact gradient method in distance in function values ***
        	PEPit guarantee:	 f(x_n)-f_* <= 0.518917 (f(x_0)-f_*)
        	Theoretical guarantee:	 f(x_n)-f_* <= 0.518917 (f(x_0)-f_*)
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    # as well as corresponding inexact gradient and function value g0 and f0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between f0 and f_*
    problem.set_initial_condition(func(x0) - fs <= 1)

    # Run n steps of the inexact gradient method
    Leps = (1 + epsilon) * L
    meps = (1 - epsilon) * mu
    gamma = 2 / (Leps + meps)

    x = x0
    for i in range(n):
        x, dx, fx = inexact_gradient_step(x, func, gamma=gamma, epsilon=epsilon, notion='relative')

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func(x) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = ((Leps - meps) / (Leps + meps)) ** (2 * n)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of inexact gradient method in distance in function values ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_inexact_gradient_descent(L=1, mu=.1, epsilon=.1, n=2,
                                                             wrapper="cvxpy", solver=None,
                                                             verbose=1)
