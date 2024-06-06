import warnings
from math import sqrt, log2

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def wc_gradient_descent_silver_stepsize_strongly_convex(L, mu, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the strongly convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu` strongly-convex.

    This code computes a worst-case guarantee for :math:`n` steps of the **gradient descent** method tuned
    according to the silver stepsize schedule.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu)` such that the guarantee

    .. math:: \\|x_n - x_\\star\\|^2 \\leqslant \\tau(n, L, \\mu) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of gradient descent using the silver stepsizes, and
    where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n`, :math:`L` and :math:`\\mu`, :math:`\\tau(n, L, \\mu)` is computed
    as the worst-case value of :math:`\\|x_n - x_\\star\\|^2` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma_t \\nabla f(x_t),

    where :math:`\\gamma_t` is a step-size of the :math:`t^{th}` step of the silver step-size schedule described in [1].

    **Theoretical guarantee**:
    The theoretical guarantee for the convergence rate of the silver stepsize can be found in [1, Theorem 4.1]:
    Let :math:`n^\\star = 2^{\\lfloor log_\\rho(L/(3\\mu)) \\rfloor}`.

    When :math:`n \\leq n^\\star`, the guarantee is given by

    .. math:: \\|x_n - x_\\star\\|^2 \\leqslant e^{-\\frac{n^{\\log_2(1 + \\sqrt{2})}}{L/\\mu}} \\|x_0-x_\\star\\|^2,

    When :math:`n > n^\\star` the guarantee is given by
    
    .. math:: \\|x_n - x_\\star\\|^2 \\leqslant e^{-\\frac{n}{n^*} \\frac{(n^*)^{\\log_2(\\rho)}}{L/\\mu}} \\|x_0-x_\\star\\|^2

    **References**:

    `[1] J. M. Altschuler, P. A. Parrilo (2023).
    Acceleration by Stepsize Hedging I: Multi-Step Descent and the Silver Stepsize Schedule.
    arXiv preprint arXiv:2309.07879.
    <https://arxiv.org/abs/2309.07879>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        n (int): number of iterations.
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
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_silver_stepsize_strongly_convex(L=3.2, mu=.1, n=8, wrapper="cvxpy", solver=None, verbose=1)
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
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.22144968332064685
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite
        		All the primal scalar constraints are verified up to an error of 1.709908798754045e-14
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 7.029501372712101e-15
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 2.0411976074102435e-12
        (PEPit) Final upper bound (dual): 0.22144968332063517 and lower bound (primal example): 0.22144968332064685 
        (PEPit) Duality gap: absolute: -1.1685097334179773e-14 and relative: -5.276637635674737e-14
        *** Example file: worst-case performance of gradient descent with silver step-sizes ***
        	PEPit guarantee:	 ||x_n - x_*||^2 <= 0.22145 ||x_0 - x_*||^2
        	Theoretical guarantee:	 ||x_n - x_*||^2 <= 0.22145 ||x_0 - x_*||^2
    
    """

    # Set n if not a power of 2
    if not log2(n).is_integer():
        warnings.warn(
            "Silver step-size strategy is optimally designed when n is a power of 2."
            " The provided input n is not a power of 2."
            " We decompose n as sum_k 2^k and recursely use sequences of stepsizes of length 2^k.")

    # Decompose n as sum of power of 2
    n_glue_list = [i for i in range(n.bit_length()) if n & (1 << i)]

    # Apply silver step-size strategy for each power of 2 composing n.
    # Initiate list of step-sizes and theoretical rate.
    h = []
    theoretical_tau = 1

    # Define a tool function
    def psi(t):
        return (1 + L / mu * t) / (1 + t)

    # Iterate over the different power of 2 composing n
    for n_glue in n_glue_list:

        # Compute 2^n_glue silver step-sizes
        y = [mu / L]
        z = [mu / L]

        a = [psi(y[0])]
        b = [psi(z[0])]

        h_temp = [b[0]]
        for step in range(n_glue):
            z_old = z[step]
            eta = 1 - z_old
            y_new = z_old / (eta + sqrt(1 + eta ** 2))
            z_new = z_old * (eta + sqrt(1 + eta ** 2))
            y.append(y_new)
            z.append(z_new)
            a_new = psi(y_new)
            b_new = psi(z_new)
            a.append(a_new)
            b.append(b_new)
            h_tilde = h_temp[:-1]
            h_temp = h_tilde + [a_new] + h_tilde + [b_new]

        # Update the list of step-sizes
        h = h + h_temp

        # Update the theoretical rate
        theoretical_tau *= ((1 - z[-1]) / (1 + z[-1])) ** 2

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the GD method
    x = x0
    for i in range(n):
        x = x - h[i] / L * func.gradient(x)

    # Set the performance metric to the distance between the output and x^*
    problem.set_performance_metric((x - xs) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent with silver step-sizes ***')
        print('\tPEPit guarantee:\t ||x_n - x_*||^2 <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||x_n - x_*||^2 <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_gradient_descent_silver_stepsize_strongly_convex(L=3.2, mu=.1, n=8,
                                                                                     wrapper="cvxpy", solver=None,
                                                                                     verbose=1)
