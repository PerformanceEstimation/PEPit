from math import sqrt

from PEPit import PEP
from PEPit.functions import SmoothConvexFunction


def wc_optimized_gradient(L, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code computes a worst-case guarantee for **optimized gradient method** (OGM). That is, it computes
    the smallest possible :math:`\\tau(n, L)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of OGM and where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n` and :math:`L`, :math:`\\tau(n, L)` is computed as the worst-case value
    of :math:`f(x_n)-f_\\star` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    The optimized gradient method is described by

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_{t+1} & = & y_t - \\frac{1}{L} \\nabla f(y_t)\\\\
                y_{t+1} & = & x_{t+1} + \\frac{\\theta_{t}-1}{\\theta_{t+1}}(x_{t+1}-x_t)+\\frac{\\theta_{t}}{\\theta_{t+1}}(x_{t+1}-y_t),
            \\end{eqnarray}

    with

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\theta_0 & = & 1 \\\\
                \\theta_t & = & \\frac{1 + \\sqrt{4 \\theta_{t-1}^2 + 1}}{2}, \\forall t \\in [|1, n-1|] \\\\
                \\theta_n & = & \\frac{1 + \\sqrt{8 \\theta_{n-1}^2 + 1}}{2}.
            \\end{eqnarray}

    **Theoretical guarantee**:
    The **tight** theoretical guarantee can be found in [2, Theorem 2]:

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{L\\|x_0-x_\\star\\|^2}{2\\theta_n^2},

    where tightness follows from [3, Theorem 3].

    **References**:
    The optimized gradient method was developed in [1, 2]; the corresponding lower bound was first obtained in [3].

    `[1] Y. Drori, M. Teboulle (2014).
    Performance of first-order methods for smooth convex minimization: a novel approach.
    Mathematical Programming 145(1–2), 451–482.
    <https://arxiv.org/pdf/1206.3209.pdf>`_

    `[2] D. Kim, J. Fessler (2016).
    Optimized first-order methods for smooth convex minimization.
    Mathematical Programming 159.1-2: 81-107.
    <https://arxiv.org/pdf/1406.5468.pdf>`_

    `[3] Y. Drori  (2017).
    The exact information-based complexity of smooth convex minimization.
    Journal of Complexity, 39, 1-16.
    <https://arxiv.org/pdf/1606.01424.pdf>`_

    Args:
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
        >>> pepit_tau, theoretical_tau = wc_optimized_gradient(L=3, n=4, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 7x7
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 30 scalar constraint(s) ...
        			Function 1 : 30 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.0767518265733206
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 4.2100544930481964e-09
        		All the primal scalar constraints are verified up to an error of 4.178030258567e-09
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 2.3578267940913163e-09
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 3.653093053290753e-08
        (PEPit) Final upper bound (dual): 0.0767518302587488 and lower bound (primal example): 0.0767518265733206 
        (PEPit) Duality gap: absolute: 3.6854281987297455e-09 and relative: 4.801746568479483e-08
        *** Example file: worst-case performance of optimized gradient method ***
        	PEPit guarantee:	 f(y_n)-f_* <= 0.0767518 ||x_0 - x_*||^2
        	Theoretical guarantee:	 f(y_n)-f_* <= 0.0767518 ||x_0 - x_*||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the optimized gradient method (OGM) method
    theta_new = 1
    x_new = x0
    y = x0
    for i in range(n):
        x_old = x_new
        x_new = y - 1 / L * func.gradient(y)
        theta_old = theta_new
        if i < n - 1:
            theta_new = (1 + sqrt(4 * theta_new ** 2 + 1)) / 2
        else:
            theta_new = (1 + sqrt(8 * theta_new ** 2 + 1)) / 2

        y = x_new + (theta_old - 1) / theta_new * (x_new - x_old) + theta_old / theta_new * (x_new - y)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func(y) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / (2 * theta_new ** 2)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of optimized gradient method ***')
        print('\tPEPit guarantee:\t f(y_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(y_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_optimized_gradient(L=3, n=4, wrapper="cvxpy", solver=None, verbose=1)
