from PEPit import PEP
from PEPit.functions.strongly_convex_function import StronglyConvexFunction


def wc_gradient_flow_strongly_convex(mu, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for a **gradient** flow.
    That is, it computes the smallest possible :math:`\\tau(\\mu)` such that the guarantee

    .. math:: \\frac{d}{dt}\\mathcal{V}(X_t) \\leqslant -\\tau(\\mu)\\mathcal{V}(X_t) ,

    is valid, where :math:`\\mathcal{V}(X_t) = f(X_t) - f(x_\\star)`, :math:`X_t` is the output of
    the **gradient** flow, and where :math:`x_\\star` is the minimizer of :math:`f`.
    In short, for given values of :math:`\\mu`, :math:`\\tau(\\mu)` is computed as the worst-case value
    of the derivative :math:`f(X_t)-f_\\star` when :math:`f(X_t) -  f(x_\\star)\\leqslant 1`.

    **Algorithm**:
    For :math:`t \\geqslant 0`,

                .. math:: \\frac{d}{dt}X_t = -\\nabla f(X_t),

    with some initialization :math:`X_{0}\\triangleq x_0`.

    **Theoretical guarantee**:

        The following **tight** guarantee can be found in [1, Proposition 11]:

        .. math:: \\frac{d}{dt}\\mathcal{V}(X_t) \\leqslant -2\\mu\\mathcal{V}(X_t).

        The detailed approach using PEPs is available in [2, Theorem 2.1].

    **References**:

    `[1] D. Scieur, V. Roulet, F. Bach and A. D'Aspremont (2017).
    Integration methods and accelerated optimization algorithms.
    In Advances in Neural Information Processing Systems (NIPS).
    <https://papers.nips.cc/paper/2017/file/bf62768ca46b6c3b5bea9515d1a1fc45-Paper.pdf>`_

    `[2] C. Moucer, A. Taylor, F. Bach (2022).
    A systematic approach to Lyapunov analyses of continuous-time models in convex optimization.
    <https://arxiv.org/pdf/2205.12772.pdf>`_

    Args:
        mu (float): the strong convexity parameter
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
        >>> pepit_tau, theoretical_tau = wc_gradient_flow_strongly_convex(mu=0.1, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 3x3
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 2 scalar constraint(s) ...
        			Function 1 : 2 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: -0.20000002010543685
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 7.21571816563555e-10
        		All the primal scalar constraints are verified up to an error of 7.074164865006338e-10
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 7.684432710751497e-09
        (PEPit) Final upper bound (dual): -0.20000002574229303 and lower bound (primal example): -0.20000002010543685 
        (PEPit) Duality gap: absolute: -5.636856176272076e-09 and relative: 2.8184278048074267e-08
        *** Example file: worst-case performance of the gradient flow ***
        	PEPit guarantee:	 d/dt[f(X_t)-f_*] <= -0.2 (f(X_t) - f(x_*))
        	Theoretical guarantee:	 d/dt[f(X_t)-f_*] <= -0.2 (f(X_t) - f(x_*))
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex function
    func = problem.declare_function(StronglyConvexFunction, mu=mu)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point xt (considering the derivative of the Lyapunov function)
    xt = problem.set_initial_point()
    gt, ft = func.oracle(xt)

    # Run the gradient flow (and define the derivative of the starting point)
    xt_dot = - gt

    # Chose the Lyapunov function and compute its derivative
    lyap = ft - fs
    lyap_dot = gt * xt_dot

    # Set the initial constraint that is a well-chosen distance between xt and x^*
    problem.set_initial_condition(lyap == 1)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(lyap_dot)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = - 2 * mu
    if mu == 0:
        print("Warning: momentum is tuned for strongly convex functions!")

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the gradient flow ***')
        print('\tPEPit guarantee:\t d/dt[f(X_t)-f_*] <= {:.6} (f(X_t) - f(x_*))'.format(pepit_tau))
        print('\tTheoretical guarantee:\t d/dt[f(X_t)-f_*] <= {:.6} (f(X_t) - f(x_*))'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_gradient_flow_strongly_convex(mu=0.1, wrapper="cvxpy", solver=None, verbose=1)
