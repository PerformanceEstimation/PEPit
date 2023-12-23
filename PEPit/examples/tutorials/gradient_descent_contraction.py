from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def wc_gradient_descent_contraction(L, mu, gamma, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for **gradient descent** with fixed step-size :math:`\\gamma`.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu, \\gamma)` such that the guarantee

    .. math:: \\| x_n - y_n \\|^2 \\leqslant \\tau(n, L, \\mu, \\gamma) \\| x_0 - y_0 \\|^2

    is valid, where :math:`x_n` and :math:`y_n` are the outputs of
    the gradient descent method with fixed step-size :math:`\\gamma`,
    starting respectively from :math:`x_0` and :math:`y_0`.

    In short, for given values of :math:`n`, :math:`L`, :math:`\\mu` and :math:`\\gamma`,
    :math:`\\tau(n, L, \\mu \\gamma)` is computed as the worst-case value of :math:`\\| x_n - y_n \\|^2`
    when :math:`\\| x_0 - y_0 \\|^2 \\leqslant 1`.

    **Algorithm**:
    For :math:`t\\in\\{0,1,\\ldots,n-1\\}`, gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma \\nabla f(x_t),

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**:
    The **tight** theoretical guarantee is

    .. math:: \\| x_n - y_n \\|^2 \\leqslant  \\max\\{(1-L\\gamma)^2,(1-\\mu \\gamma)^2\\}^n\\| x_0 - y_0 \\|^2,
    
    which is tight on simple quadratic functions.

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong-convexity parameter.
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
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_contraction(L=L, mu=0.1, gamma=1 / L, n=1, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 4x4
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 2 scalar constraint(s) ...
        			Function 1 : 2 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.8100000029203449
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 1.896548260018477e-09
        		All the primal scalar constraints are verified up to an error of 3.042855638898251e-09
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 4.0078754315331366e-08
        (PEPit) Final upper bound (dual): 0.8100000036427537 and lower bound (primal example): 0.8100000029203449 
        (PEPit) Duality gap: absolute: 7.224087994472939e-10 and relative: 8.918627121515396e-10
        *** Example file: worst-case performance of gradient descent with fixed step-sizes in contraction ***
        	PEPit guarantee:	 ||x_n - y_n||^2 <= 0.81 ||x_0 - y_0||^2
        	Theoretical guarantee:	 ||x_n - y_n||^2 <= 0.81 ||x_0 - y_0||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Then define the starting points x0 and y0 of the algorithm
    x_0 = problem.set_initial_point()
    y_0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and y0
    problem.set_initial_condition((x_0 - y_0) ** 2 <= 1)

    # Run n steps of the GD method
    x = x_0
    y = y_0
    for _ in range(n):
        x = x - gamma * func.gradient(x)
        y = y - gamma * func.gradient(y)

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric((x - y) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = max((1 - gamma * L) ** 2, (1 - gamma * mu) ** 2) ** n

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent with fixed step-sizes in contraction ***')
        print('\tPEPit guarantee:\t ||x_n - y_n||^2 <= {:.6} ||x_0 - y_0||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||x_n - y_n||^2 <= {:.6} ||x_0 - y_0||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    pepit_tau, theoretical_tau = wc_gradient_descent_contraction(L=L, mu=0.1, gamma=1 / L, n=1,
                                                                 wrapper="cvxpy", solver=None,
                                                                 verbose=1)
