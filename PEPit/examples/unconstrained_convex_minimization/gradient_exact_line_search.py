from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.primitive_steps import exact_linesearch_step


def wc_gradient_exact_line_search(L, mu, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for the **gradient descent** (GD) with **exact linesearch** (ELS).
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L, \\mu) (f(x_0) - f_\\star)

    is valid, where :math:`x_n` is the output of the GD with ELS,
    and where :math:`x_\\star` is the minimizer of :math:`f`.
    In short, for given values of :math:`n`, :math:`L` and :math:`\\mu`,
    :math:`\\tau(n, L, \\mu)` is computed as the worst-case value of
    :math:`f(x_n)-f_\\star` when :math:`f(x_0) - f_\\star \\leqslant 1`.

    **Algorithm**:
    GD with ELS can be written as

        .. math:: x_{t+1} = x_t - \\gamma_t \\nabla f(x_t)

    with :math:`\\gamma_t = \\arg\\min_{\\gamma} f \\left( x_t - \\gamma \\nabla f(x_t) \\right)`.

    **Theoretical guarantee**: The **tight** worst-case guarantee for GD with ELS, obtained in [1, Theorem 1.2], is

        .. math:: f(x_n) - f_\\star \\leqslant \\left(\\frac{L-\\mu}{L+\\mu}\\right)^{2n} (f(x_0) - f_\\star).

    **References**: The detailed approach (based on convex relaxations) is available in [1],
    along with theoretical bound.

    `[1] E. De Klerk, F. Glineur, A. Taylor (2017).
    On the worst-case complexity of the gradient method with exact line search for smooth strongly convex functions.
    Optimization Letters, 11(7), 1185-1199.
    <https://link.springer.com/content/pdf/10.1007/s11590-016-1087-4.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
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
        >>> pepit_tau, theoretical_tau = wc_gradient_exact_line_search(L=1, mu=.1, n=2, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 7x7
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 12 scalar constraint(s) ...
        			Function 1 : 12 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 1 function(s)
        			Function 1 : Adding 4 scalar constraint(s) ...
        			Function 1 : 4 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.4481249685889447
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 1.5179284576304284e-08
        		All the primal scalar constraints are verified up to an error of 1.8619346286996574e-08
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 1.3895833635451835e-07
        (PEPit) Final upper bound (dual): 0.44812496921062095 and lower bound (primal example): 0.4481249685889447 
        (PEPit) Duality gap: absolute: 6.216762660216091e-10 and relative: 1.3872832571216518e-09
        *** Example file: worst-case performance of gradient descent with exact linesearch (ELS) ***
        	PEPit guarantee:	 f(x_n)-f_* <= 0.448125 (f(x_0)-f_*)
        	Theoretical guarantee:	 f(x_n)-f_* <= 0.448125 (f(x_0)-f_*)
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value g0 and f0
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Set the initial constraint that is the difference between f0 and f_*
    problem.set_initial_condition(f0 - fs <= 1)

    # Run n steps of GD method with ELS
    x = x0
    gx = g0
    for i in range(n):
        x, gx, fx = exact_linesearch_step(x, func, [gx])

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = ((L - mu) / (L + mu)) ** (2 * n)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent with exact linesearch (ELS) ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_gradient_exact_line_search(L=1, mu=.1, n=2, wrapper="cvxpy", solver=None, verbose=1)
