from PEPit import PEP
from PEPit.functions import SmoothConvexFunction
from PEPit.primitive_steps import inexact_gradient_step


def wc_inexact_accelerated_gradient(L, epsilon, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code computes a worst-case guarantee for an **accelerated gradient method** using **inexact first-order
    information**. That is, it computes the smallest possible :math:`\\tau(n, L, \\varepsilon)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L, \\varepsilon)  \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of **inexact accelerated gradient descent** and where :math:`x_\\star`
    is a minimizer of :math:`f`.

    The inexact descent direction is assumed to satisfy a relative inaccuracy described by
    (with :math:`0\\leqslant \\varepsilon \\leqslant 1`)

    .. math:: \\|\\nabla f(y_t) - d_t\\| \\leqslant \\varepsilon \\|\\nabla f(y_t)\\|,

    where :math:`\\nabla f(y_t)` is the true gradient at :math:`y_t` and :math:`d_t` is
    the approximate descent direction that is used.

    **Algorithm**:
    The inexact accelerated gradient method of this example is provided by

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_{t+1} & = & y_t - \\frac{1}{L} d_t\\\\
                y_{k+1} & = & x_{t+1} + \\frac{t-1}{t+2} (x_{t+1} - x_t).
            \\end{eqnarray}

    **Theoretical guarantee**:
    When :math:`\\varepsilon=0`, a **tight** empirical guarantee can be found in [1, Table 1]:

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{2L\\|x_0-x_\\star\\|^2}{n^2 + 5 n + 6},

    which is achieved on some Huber loss functions (when :math:`\\varepsilon=0`).

    **References**:

    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017). Exact worst-case performance of first-order methods for composite
    convex optimization. SIAM Journal on Optimization, 27(3):1283–1313.
    <https://arxiv.org/pdf/1512.07516.pdf>`_

    Args:
        L (float): smoothness parameter.
        epsilon (float): level of inaccuracy
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
        >>> pepit_tau, theoretical_tau = wc_inexact_accelerated_gradient(L=1, epsilon=0.1, n=5, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 13x13
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 42 scalar constraint(s) ...
        			Function 1 : 42 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 1 function(s)
        			Function 1 : Adding 5 scalar constraint(s) ...
        			Function 1 : 5 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.039388047868526704
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 9.664109261839382e-09
        		All the primal scalar constraints are verified up to an error of 2.7598597017106097e-08
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 1.4089929790920282e-07
        (PEPit) Final upper bound (dual): 0.03938804252543366 and lower bound (primal example): 0.039388047868526704 
        (PEPit) Duality gap: absolute: -5.3430930443965075e-09 and relative: -1.3565264930699812e-07
        *** Example file: worst-case performance of inexact accelerated gradient method ***
        	PEPit guarantee:			 f(x_n)-f_* <= 0.039388 (f(x_0)-f_*)
        	Theoretical guarantee for epsilon = 0 :	 f(x_n)-f_* <= 0.0357143 (f(x_0)-f_*)
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothConvexFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the inexact accelerated gradient method
    x_new = x0
    y = x0
    for i in range(n):
        x_old = x_new
        x_new, dy, fy = inexact_gradient_step(y, func, gamma=1 / L, epsilon=epsilon, notion='relative')
        y = x_new + i / (i + 3) * (x_new - x_old)
    _, fx = func.oracle(x_new)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 2 * L / (n ** 2 + 5 * n + 6)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of inexact accelerated gradient method ***')
        print('\tPEPit guarantee:\t\t\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee for epsilon = 0 :\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_inexact_accelerated_gradient(L=1, epsilon=0.1, n=5,
                                                                 wrapper="cvxpy", solver=None,
                                                                 verbose=1)
