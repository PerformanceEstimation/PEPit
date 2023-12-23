from math import sqrt

from PEPit import PEP
from PEPit.functions.strongly_convex_function import StronglyConvexFunction


def wc_accelerated_gradient_flow_strongly_convex(mu, psd=True, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_{x\\in\\mathbb{R}^d} f(x),

    where :math:`f` is :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for an **accelerated gradient** flow.
    That is, it computes the smallest possible :math:`\\tau(\\mu)` such that the guarantee

    .. math:: \\frac{d}{dt}\\mathcal{V}_{P}(X_t) \\leqslant -\\tau(\\mu)\\mathcal{V}_P(X_t) ,

    is valid with 
    
    .. math:: \\mathcal{V}_{P}(X_t) = f(X_t) - f(x_\\star) + (X_t - x_\\star, \\frac{d}{dt}X_t)^T(P \\otimes I_d)(X_t - x_\\star, \\frac{d}{dt}X_t) ,
    
    where :math:`I_d` is the identity matrix, :math:`X_t` is the output of an **accelerated gradient** flow,
    and where :math:`x_\\star` is the minimizer of :math:`f`.
    
    In short, for given values of :math:`\\mu`, :math:`\\tau(\\mu)` is computed as the worst-case value of
    the derivative of :math:`f(X_t)-f_\\star` when :math:`f(X_t) -  f(x_\\star)\\leqslant 1`.

    **Algorithm**:
    For :math:`t \\geqslant 0`,

                .. math:: \\frac{d^2}{dt^2}X_t + 2\\sqrt{\\mu}\\frac{d}{dt}X_t + \\nabla f(X_t) = 0,

    with some initialization :math:`X_{0}\\triangleq x_0`.

    **Theoretical guarantee**:

        The following **tight** guarantee for :math:`P = \\frac{1}{2}\\begin{pmatrix} \\mu & \\sqrt{\\mu} \\\\ \\sqrt{\\mu} & 1\\end{pmatrix}`,
        for which :math:`\\mathcal{V}_{P} \\geqslant 0` can be found in [1, Appendix B], [2, Theorem 4.3]:

        .. math:: \\frac{d}{dt}\\mathcal{V}_P(X_t) \\leqslant -\\sqrt{\\mu}\\mathcal{V}_P(X_t).

        For :math:`P = \\begin{pmatrix} \\frac{4}{9}\\mu & \\frac{4}{3}\\sqrt{\\mu} \\\\ \\frac{4}{3}\\sqrt{\\mu} & \\frac{1}{2}\\end{pmatrix}`,
        for which :math:`\\mathcal{V}_{P}(X_t) \\geqslant 0` along the trajectory, the following **tight** guarantee can
        be found in [3, Corollary 2.5],

        .. math:: \\frac{d}{dt}\\mathcal{V}_P(X_t) \\leqslant -\\frac{4}{3}\\sqrt{\\mu}\\mathcal{V}_P(X_t).


    **References**:

    `[1] A. C. Wilson, B. Recht, M. I. Jordan (2021).
    A Lyapunov analysis of accelerated methods in optimization. In the Journal of Machine Learning Reasearch (JMLR),
    22(113):1−34, 2021.
    <https://jmlr.org/papers/volume22/20-195/20-195.pdf>`_

    `[2] J.M. Sanz-Serna and K. C. Zygalakis (2021)
    The connections between Lyapunov functions for some optimization algorithms and differential equations.
    In SIAM Journal on Numerical Analysis, 59 pp 1542-1565.
    <https://arxiv.org/pdf/2009.00673.pdf>`_

    `[3] C. Moucer, A. Taylor, F. Bach (2022).
    A systematic approach to Lyapunov analyses of continuous-time models in convex optimization.
    <https://arxiv.org/pdf/2205.12772.pdf>`_

    Args:
        mu (float): the strong convexity parameter
        psd (boolean): option for positivity of :math:`P` in the Lyapunov function :math:`\\mathcal{V}_{P}`
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
        >>> pepit_tau, theoretical_tau = wc_accelerated_gradient_flow_strongly_convex(mu=0.1, psd=True, wrapper="cvxpy", solver=None, verbose=1)
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
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: -0.31622777856752843
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite
        		All the primal scalar constraints are verified up to an error of 5.118823388165078e-16
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 5.482952099021607e-09
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 2.328561771704438e-08
        (PEPit) Final upper bound (dual): -0.31622777578996025 and lower bound (primal example): -0.31622777856752843 
        (PEPit) Duality gap: absolute: 2.7775681754604875e-09 and relative: -8.783441442249373e-09
        *** Example file: worst-case performance of an accelerated gradient flow ***
        	PEPit guarantee:	 d/dt V(X_t,t) <= -0.316228 V(X_t,t)
        	Theoretical guarantee:	 d/dt V(X_t) <= -0.316228 V(X_t,t)
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(StronglyConvexFunction, mu=mu)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point xt (considering the derivative of the Lyapunov function) and a derivative
    xt = problem.set_initial_point()
    gt, ft = func.oracle(xt)
    xt_dot = problem.set_initial_point()

    # Run the gradient flow (and define the derivative of the starting point)
    xt_dot_dot = -2 * sqrt(mu) * xt_dot - gt

    # Chose the Lyapunov function and compute its derivative
    if psd:
        lyap = ft - fs + 1 / 2 * (sqrt(mu) * (xt - xs) + xt_dot) ** 2
        lyap_dot = gt * xt_dot + (sqrt(mu) * (xt - xs) + xt_dot) * (sqrt(mu) * xt_dot + xt_dot_dot)
    else:
        lyap = ft - fs + mu * 4 / 9 * (xt - xs) ** 2 + 2 * 2 / 3 * sqrt(mu) * (xt - xs) * xt_dot + 1 / 2 * xt_dot ** 2
        lyap_dot = gt * xt_dot + mu * 8 / 9 * (xt - xs) * xt_dot + 4 / 3 * sqrt(mu) * (xt_dot ** 2 + (xt - xs) * xt_dot_dot) + xt_dot_dot * xt_dot

    # Set the initial constraint that is a well-chosen distance between xt and x_*
    problem.set_initial_condition(lyap == 1)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(lyap_dot)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    if psd:
        theoretical_tau = - sqrt(mu)
    else:
        theoretical_tau = - 4 / 3 * sqrt(mu)
    if mu == 0:
        print("Warning: momentum is tuned for strongly convex functions!")

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of an accelerated gradient flow ***')
        print('\tPEPit guarantee:\t d/dt V(X_t,t) <= {:.6} V(X_t,t)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t d/dt V(X_t) <= {:.6} V(X_t,t)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_accelerated_gradient_flow_strongly_convex(mu=0.1, psd=True,
                                                                              wrapper="cvxpy", solver=None,
                                                                              verbose=1)
