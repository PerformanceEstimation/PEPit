from math import sqrt

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def wc_robust_momentum(mu, L, lam, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

        .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly-convex.

    This code computes a worst-case guarantee for the **robust momentum method** (RMM).
    That is, it computes the smallest possible :math:`\\tau(n, \\mu, L, \\lambda)` such that the guarantee

        .. math:: v(x_{n+1}) \\leqslant \\tau(n, \\mu, L, \\lambda) v(x_{n}),

    is valid, where :math:`x_n` is the :math:`n^{\\mathrm{th}}` iterate of the RMM, and :math:`x_\\star` is a minimizer
    of :math:`f`. The function :math:`v(.)` is a well-chosen Lyapunov defined as follows,

        .. math::
            :nowrap:

            \\begin{eqnarray}
                v(x_t) & = & l\|z_t - x_\\star\|^2 + q_t, \\\\
                q_t & = & (L - \\mu) \\left(f(x_t) - f_\\star - \\frac{\\mu}{2}\|y_t - x_\\star\|^2 - \\frac{1}{2}\|\\nabla f(y_t) - \\mu (y_t - x_\\star)\|^2 \\right),
            \\end{eqnarray}

    with :math:`\\kappa = \\frac{\\mu}{L}`, :math:`\\rho = \\lambda (1 - \\frac{1}{\\kappa}) + (1 - \\lambda) \\left(1 - \\frac{1}{\\sqrt{\\kappa}}\\right)`, and :math:`l = \\mu^2  \\frac{\\kappa - \\kappa \\rho^2 - 1}{2 \\rho (1 - \\rho)}``.

    **Algorithm**:

    For :math:`t \\in \\{0, \\dots, n-1\\}`,

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_{t+1} & = & x_{t} + \\beta (x_t - x_{t-1}) - \\alpha \\nabla f(y_t), \\\\
                y_{t+1} & = & y_{t} + \\gamma (x_t - x_{t-1}),
            \\end{eqnarray}

    with :math:`x_{-1}, x_0 \\in \\mathrm{R}^d`,
    and with parameters :math:`\\alpha = \\frac{\\kappa (1 - \\rho^2)(1 + \\rho)}{L}`, :math:`\\beta = \\frac{\\kappa \\rho^3}{\\kappa - 1}`, :math:`\\gamma = \\frac{\\rho^2}{(\\kappa - 1)(1 - \\rho)^2(1 + \\rho)}`.
    
    **Theoretical guarantee**:

    A convergence guarantee (empirically tight) is obtained in [1, Theorem 1],
    
        .. math:: v(x_{n+1}) \\leqslant \\rho^2 v(x_n),

    with :math:`\\rho = \\lambda (1 - \\frac{1}{\\kappa}) + (1 - \\lambda) \\left(1 - \\frac{1}{\\sqrt{\\kappa}}\\right)`.

    **References**:

    `[1] S. Cyrus, B. Hu, B. Van Scoy, L. Lessard (2018).
    A robust accelerated optimization algorithm for strongly convex functions.
    American Control Conference (ACC).
    <https://arxiv.org/pdf/1710.04753.pdf>`_
         
    Args:    
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        lam (float): if :math:`\\lambda=1` it is the gradient descent, if :math:`\\lambda=0`,
                     it is the Triple Momentum Method.
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
    
    Examples:
        >>> pepit_tau, theoretical_tau = wc_robust_momentum(mu=0.1, L=1, lam=0.2, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 5x5
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 6 scalar constraint(s) ...
        			Function 1 : 6 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.5285548454743232
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite
        		All the primal scalar constraints are verified
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 6.797521527290181e-08
        (PEPit) Final upper bound (dual): 0.5285548474610776 and lower bound (primal example): 0.5285548454743232 
        (PEPit) Duality gap: absolute: 1.9867544276408466e-09 and relative: 3.758842520605294e-09
        *** Example file: worst-case performance of the Robust Momentum Method ***
        	PEPit guarantee:	 v(x_(n+1)) <= 0.528555 v(x_n)
        	Theoretical guarantee:	 v(x_(n+1)) <= 0.528555 v(x_n)
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then Define the starting points of the algorithm, x0 and x1
    x0 = problem.set_initial_point()
    x1 = problem.set_initial_point()

    # Set the parameters of the robust momentum method
    kappa = L / mu
    rho = lam * (1 - 1 / kappa) + (1 - lam) * (1 - 1 / sqrt(kappa))
    alpha = kappa * (1 - rho) ** 2 * (1 + rho) / L
    beta = kappa * rho ** 3 / (kappa - 1)
    gamma = rho ** 3 / ((kappa - 1) * (1 - rho) ** 2 * (1 + rho))
    l = mu ** 2 * (kappa - kappa * rho ** 2 - 1) / (2 * rho * (1 - rho))

    # Run one step of the Robust Momentum Method
    y0 = x1 + gamma * (x1 - x0)
    g0, f0 = func.oracle(y0)
    x2 = x1 + beta * (x1 - x0) - alpha * g0
    y1 = x2 + gamma * (x2 - x1)
    g1, f1 = func.oracle(y1)
    x3 = x2 + beta * (x2 - x1) - alpha * g1

    z1 = (x2 - (rho ** 2) * x1) / (1 - rho ** 2)
    z2 = (x3 - (rho ** 2) * x2) / (1 - rho ** 2)

    # Evaluate the lyapunov function at the first and second iterates
    q0 = (L - mu) * (f0 - fs - mu / 2 * (y0 - xs) ** 2) - 1 / 2 * (g0 - mu * (y0 - xs)) ** 2
    q1 = (L - mu) * (f1 - fs - mu / 2 * (y1 - xs) ** 2) - 1 / 2 * (g1 - mu * (y1 - xs)) ** 2
    initLyapunov = l * (z1 - xs) ** 2 + q0
    finalLyapunov = l * (z2 - xs) ** 2 + q1

    # Set the initial constraint that is a bound on the initial Lyapunov function
    problem.set_initial_condition(initLyapunov <= 1)

    # Set the performance metric to the final distance to optimum, that is the final Lyapunov function
    problem.set_performance_metric(finalLyapunov)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = rho ** 2

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the Robust Momentum Method ***')
        print('\tPEPit guarantee:\t v(x_(n+1)) <= {:.6} v(x_n)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t v(x_(n+1)) <= {:.6} v(x_n)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_robust_momentum(mu=0.1, L=1, lam=0.2, wrapper="cvxpy", solver=None, verbose=1)
