from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.primitive_steps import exact_linesearch_step
from PEPit.primitive_steps import inexact_gradient_step


def wc_inexact_gradient_exact_line_search(L, mu, epsilon, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

        .. math:: f_\\star \\triangleq \min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for an **inexact gradient method with exact linesearch (ELS)**.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu, \\varepsilon)` such that the guarantee

        .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L, \\mu, \\varepsilon) ( f(x_0) - f_\\star )

    is valid, where :math:`x_n` is the output of the **gradient descent with an inexact descent direction
    and an exact linesearch**, and where :math:`x_\\star` is the minimizer of :math:`f`.

    The inexact descent direction :math:`d` is assumed to satisfy a relative inaccuracy described by
    (with :math:`0 \\leqslant \\varepsilon < 1`)

        .. math:: \\|\\nabla f(x_t) - d_t\\| \\leqslant \\varepsilon \\|\\nabla f(x_t)\\|,

    where :math:`\\nabla f(x_t)` is the true gradient, and :math:`d_t`
    is the approximate descent direction that is used.

    **Algorithm**:

    For :math:`t \\in \\{0, \\dots, n-1\\}`,

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\gamma_t & = & \\arg\\min_{\\gamma \in R^d} f(x_t- \\gamma d_t), \\\\
                x_{t+1} & = & x_t - \\gamma_t d_t.
            \\end{eqnarray}

    **Theoretical guarantees**:

    The **tight** guarantee obtained in [1, Theorem 5.1] is

        .. math:: f(x_n) - f_\\star\\leqslant \\left(\\frac{L_{\\varepsilon} - \\mu_{\\varepsilon}}{L_{\\varepsilon} + \\mu_{\\varepsilon}}\\right)^{2n}( f(x_0) - f_\\star ),

    with :math:`L_{\\varepsilon} = (1 + \\varepsilon) L` and :math:`\\mu_{\\varepsilon} = (1 - \\varepsilon) \\mu`.
    Tightness is achieved on simple quadratic functions.

    **References**: The detailed approach (based on convex relaxations) is available in [1],

    `[1]  E. De Klerk, F. Glineur, A. Taylor (2017). On the worst-case complexity of the gradient method with exact
    line search for smooth strongly convex functions. Optimization Letters, 11(7), 1185-1199.
    <https://link.springer.com/content/pdf/10.1007/s11590-016-1087-4.pdf>`_

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
        >>> pepit_tau, theoretical_tau = wc_inexact_gradient_exact_line_search(L=1, mu=0.1, epsilon=0.1, n=2, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 9x9
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 12 scalar constraint(s) ...
        			Function 1 : 12 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 1 function(s)
        			Function 1 : Adding 6 scalar constraint(s) ...
        			Function 1 : 6 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.5189166579925197
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 7.855711098683355e-09
        		All the primal scalar constraints are verified up to an error of 2.1750143269771982e-08
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 8.345007454521682e-08
        (PEPit) Final upper bound (dual): 0.5189166453888967 and lower bound (primal example): 0.5189166579925197 
        (PEPit) Duality gap: absolute: -1.2603623034124212e-08 and relative: -2.428833771280839e-08
        *** Example file: worst-case performance of inexact gradient descent with exact linesearch ***
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

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value g0 and f0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between f0 and f_*
    problem.set_initial_condition(func(x0) - fs <= 1)

    # Run n steps of the inexact gradient method with ELS
    x = x0
    for i in range(n):
        _, dx, _ = inexact_gradient_step(x, func, gamma=0, epsilon=epsilon, notion='relative')
        x, gx, fx = exact_linesearch_step(x, func, [dx])

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func(x) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    Leps = (1 + epsilon) * L
    meps = (1 - epsilon) * mu
    theoretical_tau = ((Leps - meps) / (Leps + meps)) ** (2 * n)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of inexact gradient descent with exact linesearch ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_inexact_gradient_exact_line_search(L=1, mu=0.1, epsilon=0.1, n=2,
                                                                       wrapper="cvxpy", solver=None,
                                                                       verbose=1)
