from math import sqrt

from PEPit import PEP
from PEPit.functions import ConvexFunction
from PEPit.primitive_steps import proximal_step


def wc_accelerated_proximal_point(A0, gammas, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is  convex and possibly non-smooth.

    This code computes a worst-case guarantee an **accelerated proximal point** method,
    aka **fast proximal point** method (FPP).
    That is, it computes the smallest possible :math:`\\tau(n, A_0,\\vec{\\gamma})` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, A_0, \\vec{\\gamma}) \\left(f(x_0) - f_\\star + \\frac{A_0}{2}  \\|x_0 - x_\\star\\|^2\\right)

    is valid, where :math:`x_n` is the output of FPP (with step-size :math:`\\gamma_t` at step
    :math:`t\\in \\{0, \\dots, n-1\\}`) and where :math:`x_\\star` is a minimizer of :math:`f`
    and :math:`A_0` is a positive number.

    In short, for given values of :math:`n`,  :math:`A_0` and :math:`\\vec{\\gamma}`, :math:`\\tau(n)`
    is computed as the worst-case value of :math:`f(x_n)-f_\\star`
    when :math:`f(x_0) - f_\\star + \\frac{A_0}{2} \\|x_0 - x_\\star\\|^2 \\leqslant 1`, for the following method.

    **Algorithm**:
    For :math:`t\\in \\{0, \\dots, n-1\\}`:

       .. math::
           :nowrap:

           \\begin{eqnarray}
               y_{t+1} & = & (1-\\alpha_{t} ) x_{t} + \\alpha_{t} v_t \\\\
               x_{t+1} & = & \\arg\\min_x \\left\\{f(x)+\\frac{1}{2\\gamma_t}\\|x-y_{t+1}\\|^2 \\right\\}, \\\\
               v_{t+1} & = & v_t + \\frac{1}{\\alpha_{t}} (x_{t+1}-y_{t+1})
           \\end{eqnarray}

    with

       .. math::
           :nowrap:

           \\begin{eqnarray}
               \\alpha_{t} & = & \\frac{\\sqrt{(A_t \\gamma_t)^2 + 4 A_t \\gamma_t} - A_t \\gamma_t}{2} \\\\
               A_{t+1} & = & (1 - \\alpha_{t}) A_t
           \\end{eqnarray}

    and :math:`v_0=x_0`.



    **Theoretical guarantee**:
    A theoretical **upper** bound can be found in [1, Theorem 2.3.]:

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{4}{A_0 (\\sum_{t=0}^{n-1} \\sqrt{\\gamma_t})^2}\\left(f(x_0) - f_\\star + \\frac{A_0}{2}  \\|x_0 - x_\\star\\|^2 \\right).

    **References**:
    The accelerated proximal point was first obtained and analyzed in [1].

    `[1] O. Güler (1992).
    New proximal point algorithms for convex minimization.
    SIAM Journal on Optimization, 2(4):649–664.
    <https://epubs.siam.org/doi/abs/10.1137/0802032?mobileUi=0>`_


    Args:
        A0 (float): initial value for parameter A_0.
        gammas (list): sequence of step-sizes.
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
        >>> pepit_tau, theoretical_tau = wc_accelerated_proximal_point(A0=5, gammas=[(i + 1) / 1.1 for i in range(3)], n=3, wrapper="cvxpy", solver=None, verbose=1)
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
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.015931148923290624
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 3.713119626105772e-10
        		All the primal scalar constraints are verified up to an error of 1.4460231649235378e-09
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 2.0490523713620816e-10
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 2.9451787884841313e-09
        (PEPit) Final upper bound (dual): 0.015931149263944334 and lower bound (primal example): 0.015931148923290624 
        (PEPit) Duality gap: absolute: 3.4065371010139067e-10 and relative: 2.138287148915985e-08
        *** Example file: worst-case performance of fast proximal point method ***
        	PEPit guarantee:	 f(x_n)-f_* <= 0.0159311 (f(x_0) - f_* + A/2* ||x_0 - x_*||^2)
        	Theoretical guarantee:	 f(x_n)-f_* <= 0.0511881 (f(x_0) - f_* + A/2* ||x_0 - x_*||^2)
    
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

    # Set the initial constraint that is a well-chosen distance between x0 and x^*
    problem.set_initial_condition(func(x0) - fs + A0 / 2 * (x0 - xs) ** 2 <= 1)

    # Run the fast proximal point method
    x, v = x0, x0
    A = A0
    for i in range(n):
        alpha = (sqrt((A * gammas[i]) ** 2 + 4 * A * gammas[i]) - A * gammas[i]) / 2
        y = (1 - alpha) * x + alpha * v
        x, _, _ = proximal_step(y, func, gammas[i])
        v = v + 1 / alpha * (x - y)
        A = (1 - alpha) * A

    # Set the performance metric to the final distance to optimum in function values
    problem.set_performance_metric(func(x) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    accumulation = 0
    for i in range(n):
        accumulation += sqrt(gammas[i])
    theoretical_tau = 4 / A0 / accumulation ** 2

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of fast proximal point method ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0) - f_* + A/2* ||x_0 - x_*||^2)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0) - f_* + A/2* ||x_0 - x_*||^2)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_accelerated_proximal_point(A0=5, gammas=[(i + 1) / 1.1 for i in range(3)], n=3,
                                                               wrapper="cvxpy", solver=None,
                                                               verbose=1)
