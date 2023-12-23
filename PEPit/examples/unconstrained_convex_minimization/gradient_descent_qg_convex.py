from PEPit import PEP
from PEPit.functions import ConvexQGFunction


def wc_gradient_descent_qg_convex(L, gamma, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is quadratically upper bounded (:math:`\\text{QG}^+` [1]), i.e.
    :math:`\\forall x, f(x) - f_\\star \\leqslant \\frac{L}{2} \\|x-x_\\star\\|^2`, and convex.

    This code computes a worst-case guarantee for **gradient descent** with fixed step-size :math:`\\gamma`.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\gamma)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L, \\gamma) \\| x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of gradient descent with fixed step-size :math:`\\gamma`, and
    where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n`, :math:`L`,
    and :math:`\\gamma`, :math:`\\tau(n, L, \\gamma)` is computed as the worst-case
    value of :math:`f(x_n)-f_\\star` when :math:`||x_0 - x_\\star||^2 \\leqslant 1`.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma \\nabla f(x_t),

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**:
    When :math:`\\gamma < \\frac{1}{L}`, the **lower** theoretical guarantee can be found in [1, Theorem 2.2]:

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{L}{2}\\max\\left(\\frac{1}{2n L \\gamma + 1}, L \\gamma\\right) \\|x_0-x_\\star\\|^2.

    **References**:

    The detailed approach is available in [1, Theorem 2.2].

    `[1] B. Goujaud, A. Taylor, A. Dieuleveut (2022).
    Optimal first-order methods for convex functions with a quadratic upper bound.
    <https://arxiv.org/pdf/2205.15033.pdf>`_

    Args:
        L (float): the quadratic growth parameter.
        gamma (float): step-size.
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
        >>> L = 1
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_qg_convex(L=L, gamma=.2 / L, n=4, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 7x7
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 35 scalar constraint(s) ...
        			Function 1 : 35 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.19230769057979225
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite
        		All the primal scalar constraints are verified up to an error of 6.487419074163725e-10
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 9.613757534126084e-10
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 9.03711530175072e-09
        (PEPit) Final upper bound (dual): 0.1923076915693468 and lower bound (primal example): 0.19230769057979225 
        (PEPit) Duality gap: absolute: 9.895545494131852e-10 and relative: 5.145683703182944e-09
        *** Example file: worst-case performance of gradient descent with fixed step-sizes ***
        	PEPit guarantee:	 f(x_n)-f_* <= 0.192308 ||x_0 - x_*||^2
        	Theoretical guarantee:	 f(x_n)-f_* <= 0.192308 ||x_0 - x_*||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(ConvexQGFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the GD method
    x = x0
    for i in range(n):
        x = x - gamma * func.gradient(x)

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric(func.value(x) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / 2 * max(1 / (2 * n * L * gamma + 1), L * gamma)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent with fixed step-sizes ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    pepit_tau, theoretical_tau = wc_gradient_descent_qg_convex(L=L, gamma=.2 / L, n=4,
                                                               wrapper="cvxpy", solver=None,
                                                               verbose=1)
