import warnings
from math import sqrt, log2

from PEPit import PEP
from PEPit.functions import SmoothConvexFunction


def wc_gradient_descent_silver_stepsize_convex(L, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code computes a worst-case guarantee for :math:`n` steps of the **gradient descent** method tuned
    according to the silver stepsize schedule.
    That is, it computes the smallest possible :math:`\\tau(n, L)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of gradient descent using the silver step-sizes, and
    where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n`, and :math:`L`, :math:`\\tau(n, L)` is computed as the worst-case
    value of :math:`f(x_n)-f_\\star` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma_t \\nabla f(x_t),

    where :math:`\\gamma_t` is a step-size of the :math:`t^{th}` step of the silver step-size schedule described in [1].

    **Theoretical guarantee**:
    The theoretical guarantee for the convergence rate of the silver stepsize can be found in [1, Theorem 1.1]:

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{L}{1 + \\sqrt{4(1 + \\sqrt{2})^{2k}-3}} \\|x_0-x_\\star\\|^2,

    where :math:`k` is such that :math:`n = 2^k - 1`.

    **References**:

    `[1] J. M. Altschuler, P. A. Parrilo (2023).
    Acceleration by Stepsize Hedging II: Silver Stepsize Schedule for Smooth Convex Optimization.
    arXiv preprint arXiv:2309.16530.
    <https://arxiv.org/abs/2309.16530>`_

    Args:
        L (float): the smoothness parameter.
        n (int): number of iterations (will be reset to the largest power of 2 minus 1 smaller than the provided value).
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): Level of information details to print.

                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_silver_stepsize_convex(L=10, n=7, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 10x10
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 72 scalar constraint(s) ...
        			Function 1 : 72 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.1842154142304195
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 2.9723718334693814e-10
        		All the primal scalar constraints are verified up to an error of 2.326594000789939e-09
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 2.421699121665203e-09
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 3.163353475812893e-08
        (PEPit) Final upper bound (dual): 0.18421541696277668 and lower bound (primal example): 0.1842154142304195 
        (PEPit) Duality gap: absolute: 2.732357173851341e-09 and relative: 1.4832402517813552e-08
        *** Example file: worst-case performance of gradient descent with silver step-sizes ***
        	PEPit guarantee:	 f(x_n)-f_* <= 0.184215 ||x_0 - x_*||^2
        	Theoretical guarantee:	 f(x_n)-f_* <= 0.343775 ||x_0 - x_*||^2
    
    """

    # Modify n if not a power of 2 minus 1
    k = log2(n + 1)
    if not k.is_integer():
        warnings.warn("Silver step-size strategy is only defined when n is a power of 2 minus 1."
                      " The provided input n is not a power of 2 minus 1."
                      " n is reset as the largest acceptable value smaller than the provided one.")
        k = int(k)
        n = 2 ** k - 1

    # Define silver stepsizes
    def fast_dyadic_valuation(i):
        return (i & -i).bit_length() - 1
    h = [1 + (1 + sqrt(2)) ** (fast_dyadic_valuation(i)-1) for i in range(1, n + 1)]

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

    # Run n steps of the GD method
    x = x0
    for i in range(n):
        x = x - h[i] / L * func.gradient(x)

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric(func(x) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / (1 + sqrt(4 * (1 + sqrt(2)) ** (2 * k) - 3))

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent with silver step-sizes ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_gradient_descent_silver_stepsize_convex(L=10, n=7,
                                                                            wrapper="cvxpy", solver=None, verbose=1)
