from PEPit import PEP
from PEPit.functions import ConvexFunction
from PEPit.primitive_steps import proximal_step


def wc_proximal_point(gamma, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is closed, proper, and convex (and potentially non-smooth).

    This code computes a worst-case guarantee for the **proximal point method** with step-size :math:`\\gamma`.
    That is, it computes the smallest possible :math:`\\tau(n,\\gamma)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, \\gamma)  \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the proximal point method, and where :math:`x_\\star` is a
    minimizer of :math:`f`.

    In short, for given values of :math:`n` and :math:`\\gamma`,
    :math:`\\tau(n,\\gamma)` is computed as the worst-case value of :math:`f(x_n)-f_\\star`
    when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:

    The proximal point method is described by

        .. math:: x_{t+1} = \\arg\\min_x \\left\\{f(x)+\\frac{1}{2\gamma}\\|x-x_t\\|^2 \\right\\},

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**:

    The **tight** theoretical guarantee can be found in [1, Theorem 4.1]:

        .. math:: f(x_n)-f_\\star \\leqslant \\frac{\\|x_0-x_\\star\\|^2}{4\\gamma n},

    where tightness is obtained on, e.g., one-dimensional linear problems on the positive orthant.

    **References**:

    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
    Exact worst-case performance of first-order methods for composite convex optimization.
    SIAM Journal on Optimization, 27(3):1283–1313.
    <https://arxiv.org/pdf/1512.07516.pdf>`_

    Args:
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
        >>> pepit_tau, theoretical_tau = wc_proximal_point(gamma=3, n=4, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 6x6
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 20 scalar constraint(s) ...
        			Function 1 : 20 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.020833335685730252
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 3.626659005644299e-09
        		All the primal scalar constraints are verified up to an error of 1.1386158081487519e-08
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 3.0464297827437203e-08
        (PEPit) Final upper bound (dual): 0.020833337068527292 and lower bound (primal example): 0.020833335685730252 
        (PEPit) Duality gap: absolute: 1.382797040067052e-09 and relative: 6.637425042856655e-08
        *** Example file: worst-case performance of proximal point method ***
        	PEPit guarantee:	 f(x_n)-f_* <= 0.0208333 ||x_0 - x_*||^2
        	Theoretical guarantee:	 f(x_n)-f_* <= 0.0208333 ||x_0 - x_*||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function
    func = problem.declare_function(ConvexFunction)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the proximal point method
    x = x0
    for _ in range(n):
        x, _, fx = proximal_step(x, func, gamma)

    # Set the performance metric to the final distance to optimum in function values
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 1 / (4 * gamma * n)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of proximal point method ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_proximal_point(gamma=3, n=4, wrapper="cvxpy", solver=None, verbose=1)
