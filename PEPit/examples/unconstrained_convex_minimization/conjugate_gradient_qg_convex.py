from PEPit import PEP
from PEPit.functions.convex_qg_function import ConvexQGFunction
from PEPit.primitive_steps import exact_linesearch_step


def wc_conjugate_gradient_qg_convex(L, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is quadratically upper bounded (:math:`\\text{QG}^+` [2]), i.e.
    :math:`\\forall x, f(x) - f_\\star \\leqslant \\frac{L}{2} \\|x-x_\\star\\|^2`, and convex.

    This code computes a worst-case guarantee for the **conjugate gradient (CG)** method (with exact span searches).
    That is, it computes the smallest possible :math:`\\tau(n, L)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L) \\|x_0-x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the **conjugate gradient** method,
    and where :math:`x_\\star` is a minimizer of :math:`f`.
    In short, for given values of :math:`n` and :math:`L`,
    :math:`\\tau(n, L)` is computed as the worst-case value of
    :math:`f(x_n)-f_\\star` when :math:`\\|x_0-x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:

        .. math:: x_{t+1} = x_t - \\sum_{i=0}^t \\gamma_i \\nabla f(x_i)

        with

        .. math:: (\\gamma_i)_{i \\leqslant t} = \\arg\\min_{(\\gamma_i)_{i \\leqslant t}} f \\left(x_t - \\sum_{i=0}^t \\gamma_i \\nabla f(x_i) \\right)

    **Theoretical guarantee**:

        The **tight** guarantee obtained in [2, Theorem 2.3] (lower) and [2, Theorem 2.4] (upper) is

        .. math:: f(x_n) - f_\\star \\leqslant \\frac{L}{2 (n + 1)} \\|x_0-x_\\star\\|^2.

    **References**:
    The detailed approach (based on convex relaxations) is available in [1, Corollary 6],
    and the result provided in [2, Theorem 2.4].

    `[1] Y. Drori and A. Taylor (2020). Efficient first-order methods for convex minimization: a constructive approach.
    Mathematical Programming 184 (1), 183-220.
    <https://arxiv.org/pdf/1803.05676.pdf>`_

    `[2] B. Goujaud, A. Taylor, A. Dieuleveut (2022).
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
        >>> pepit_tau, theoretical_tau = wc_conjugate_gradient_qg_convex(L=1, n=12, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 27x27
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 195 scalar constraint(s) ...
        			Function 1 : 195 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 1 function(s)
        			Function 1 : Adding 156 scalar constraint(s) ...
        			Function 1 : 156 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.038461537777152104
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 9.102635033370141e-10
        		All the primal scalar constraints are verified up to an error of 6.985771551504261e-09
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 6.487940056145134e-09
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 2.3524476901692115e-07
        (PEPit) Final upper bound (dual): 0.038461545907145095 and lower bound (primal example): 0.038461537777152104 
        (PEPit) Duality gap: absolute: 8.129992991323665e-09 and relative: 2.113798215357174e-07
        *** Example file: worst-case performance of conjugate gradient method ***
        	PEPit guarantee:	 f(x_n)-f_* <= 0.0384615 ||x_0 - x_*||^2
        	Theoretical guarantee:	 f(x_n)-f_* <= 0.0384615 ||x_0 - x_*||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(ConvexQGFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x_*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the Conjugate Gradient method
    x_new = x0
    g0, f0 = func.oracle(x0)
    span = [g0]  # list of search directions
    for i in range(n):
        x_old = x_new
        x_new, gx, fx = exact_linesearch_step(x_new, func, span)
        span.append(gx)
        span.append(x_old - x_new)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / (2 * (n + 1))

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of conjugate gradient method ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_conjugate_gradient_qg_convex(L=1, n=12, wrapper="cvxpy", solver=None, verbose=1)
