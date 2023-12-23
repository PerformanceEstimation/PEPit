from math import sqrt

from PEPit import PEP
from PEPit.functions import ConvexQGFunction


def wc_gradient_descent_qg_convex_decreasing(L, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is quadratically upper bounded (:math:`\\text{QG}^+` [1]), i.e.
    :math:`\\forall x, f(x) - f_\\star \\leqslant \\frac{L}{2} \\|x-x_\\star\\|^2`, and convex.

    This code computes a worst-case guarantee for **gradient descent** with decreasing step-sizes.
    That is, it computes the smallest possible :math:`\\tau(n, L)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L) \\| x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of gradient descent with decreasing step-sizes, and
    where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n` and :math:`L`,
    :math:`\\tau(n, L)` is computed as the worst-case
    value of :math:`f(x_n)-f_\\star` when :math:`||x_0 - x_\\star||^2 \\leqslant 1`.

    **Algorithm**:
    Gradient descent with decreasing step sizes is described by

    .. math:: x_{t+1} = x_t - \\gamma_t \\nabla f(x_t)

    with

    .. math:: \\gamma_t = \\frac{1}{L u_{t+1}}

    where the sequence :math:`u` is defined by

    .. math::
        :nowrap:

        \\begin{eqnarray}
            u_0 & = & 1 \\\\
            u_{t} & = & \\frac{u_{t-1}}{2} + \\sqrt{\\left(\\frac{u_{t-1}}{2}\\right)^2 + 2}, \\quad \\mathrm{for } t \\geq 1
        \\end{eqnarray}

    **Theoretical guarantee**:
    The **tight** theoretical guarantee is conjectured in [1, Conjecture A.3]:

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{L}{2 u_t} \\|x_0-x_\\star\\|^2.

    Notes:

    We verify that :math:`u_t \\sim 2\\sqrt{t}`.
    The step sizes as well as the function values of the iterates decrease as
    :math:`O\\left( \\frac{1}{\\sqrt{t}} \\right)`.

    **References**:

    The detailed approach is available in [1, Appendix A.3].

    `[1] B. Goujaud, A. Taylor, A. Dieuleveut (2022).
    Optimal first-order methods for convex functions with a quadratic upper bound.
    <https://arxiv.org/pdf/2205.15033.pdf>`_

    Args:
        L (float): the quadratic growth parameter.
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
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_qg_convex_decreasing(L=1, n=6, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 9x9
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 63 scalar constraint(s) ...
        			Function 1 : 63 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.10554738683923168
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 1.3052503007966848e-10
        		All the primal scalar constraints are verified up to an error of 6.537865110400887e-10
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 4.781039101724198e-10
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 8.076454220612452e-09
        (PEPit) Final upper bound (dual): 0.10554738755645543 and lower bound (primal example): 0.10554738683923168 
        (PEPit) Duality gap: absolute: 7.172237526109626e-10 and relative: 6.7952772123428116e-09
        *** Example file: worst-case performance of gradient descent with fixed step-sizes ***
        	PEPit guarantee:	 f(x_n)-f_* <= 0.105547 ||x_0 - x_*||^2
        	Theoretical conjecture:	 f(x_n)-f_* <= 0.105547 ||x_0 - x_*||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(ConvexQGFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x = problem.set_initial_point()
    g, f = func.oracle(x)

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x - xs) ** 2 <= 1)

    # GD loop
    u = 1
    for i in range(n):
        # Run 1 step of the GD method and update u accordingly.
        u = u / 2 + sqrt((u / 2) ** 2 + 2)
        gamma = 1 / (L * u)
        x = x - gamma * g
        g, f = func.oracle(x)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / (2 * u)

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric((f - fs))

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent with fixed step-sizes ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical conjecture:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_gradient_descent_qg_convex_decreasing(L=1, n=6,
                                                                          wrapper="cvxpy", solver=None,
                                                                          verbose=1)
