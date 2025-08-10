from math import sqrt

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def wc_accelerated_gradient_convex(mu, L, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex (:math:`\\mu` is possibly 0).

    This code computes a worst-case guarantee for an **accelerated gradient method**, a.k.a. **fast gradient method** [1].
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L, \\mu) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the accelerated gradient method,
    and where :math:`x_\\star` is the minimizer of :math:`f`.
    In short, for given values of :math:`n`, :math:`L` and :math:`\\mu`,
    :math:`\\tau(n, L, \\mu)` is computed as the worst-case value of
    :math:`f(x_n)-f_\\star` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**: Initialize :math:`\\lambda_1=1`, :math:`y_1=x_0`.
    One iteration of accelerated gradient method is described by

    .. math::

        \\begin{eqnarray}
            \\text{Set: }\\lambda_{t+1} & = & \\frac{1 + \\sqrt{4\\lambda_t^2 + 1}}{2} \\\\
            x_{t} & = & y_t - \\frac{1}{L} \\nabla f(y_t),\\\\
            y_{t+1} & = & x_{t} + \\frac{\\lambda_t-1}{\\lambda_{t+1}} (x_t-x_{t-1}).
        \\end{eqnarray}

    **Theoretical guarantee**: The following worst-case guarantee can be found in e.g., [2, Theorem 4.4]:

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{L}{2}\\frac{\\|x_0-x_\\star\\|^2}{\\lambda_n^2}.

    **References**:
    
    `[1] Y. Nesterov (1983).
    A method for solving the convex programming problem with convergence rate O(1/k^2).
    In Dokl. akad. nauk Sssr (Vol. 269, pp. 543-547).
    <http://www.mathnet.ru/links/9bcb158ed2df3d8db3532aafd551967d/dan46009.pdf>`_
    
    `[2] A. Beck, M. Teboulle (2009).
    A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.
    SIAM journal on imaging sciences, 2009, vol. 2, no 1, p. 183-202.
    <https://www.ceremade.dauphine.fr/~carlier/FISTA>`_

    Args:
        mu (float): the strong convexity parameter
        L (float): the smoothness parameter.
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
        >>> pepit_tau, theoretical_tau = wc_accelerated_gradient_convex(mu=0, L=1, n=1, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 4x4
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 6 scalar constraint(s) ...
        			Function 1 : 6 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.16666666115098375
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 4.82087966328108e-09
        		All the primal scalar constraints are verified up to an error of 3.6200406144937247e-09
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 3.101096412994053e-08
        (PEPit) Final upper bound (dual): 0.16666666498582347 and lower bound (primal example): 0.16666666115098375 
        (PEPit) Duality gap: absolute: 3.834839723548811e-09 and relative: 2.3009039102756247e-08
        *** Example file: worst-case performance of accelerated gradient method ***
        	PEPit guarantee:	 f(x_n)-f_* <= 0.166667 ||x_0 - x_*||^2
        	Theoretical guarantee:	 f(x_n)-f_* <= 0.5 ||x_0 - x_*||^2
    
    """
    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the fast gradient method
    x = x0
    y = x0
    lam = 1
    
    for _ in range(n):
        lam_old = lam
        lam = (1 + sqrt(4 * lam_old ** 2 + 1)) / 2
        x_old = x
        x = y - 1 / L * func.gradient(y)
        y = x + (lam_old - 1) / lam * (x - x_old)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func(x) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Theoretical guarantee (for comparison)
    theoretical_tau = L / (2 * lam_old**2)
    
    if mu != 0:
        print('Warning: momentum is tuned for non-strongly convex functions.')

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of accelerated gradient method ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_accelerated_gradient_convex(mu=0, L=1, n=1, wrapper="cvxpy",
                                                                solver=None, verbose=1)
