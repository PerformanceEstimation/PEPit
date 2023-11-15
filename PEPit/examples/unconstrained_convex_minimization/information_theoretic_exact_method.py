from math import sqrt

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def wc_information_theoretic(mu, L, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex (:math:`\\mu` is possibly 0).

    This code computes a worst-case guarantee for the **information theoretic exact method** (ITEM).
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu)` such that the guarantee

    .. math:: \\|z_n - x_\\star\\|^2 \\leqslant \\tau(n, L, \\mu) \\|z_0 - x_\\star\\|^2

    is valid, where :math:`z_n` is the output of the ITEM,
    and where :math:`x_\\star` is the minimizer of :math:`f`.
    In short, for given values of :math:`n`, :math:`L` and :math:`\\mu`,
    :math:`\\tau(n, L, \\mu)` is computed as the worst-case value of
    :math:`\\|z_n - x_\\star\\|^2` when :math:`\\|z_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    For :math:`t\\in\\{0,1,\\ldots,n-1\\}`, the information theoretic exact method of this example is provided by

        .. math::
            :nowrap:

            \\begin{eqnarray}
                y_{t} & = & (1-\\beta_t) z_t + \\beta_t x_t \\\\
                x_{t+1} & = & y_t - \\frac{1}{L} \\nabla f(y_t) \\\\
                z_{t+1} & = & \\left(1-q\\delta_t\\right) z_t+q\\delta_t y_t-\\frac{\\delta_t}{L}\\nabla f(y_t),
            \\end{eqnarray}

    with :math:`y_{-1}=x_0=z_0`, :math:`q=\\frac{\\mu}{L}` (inverse condition ratio), and the scalar sequences:

        .. math::
            :nowrap:

            \\begin{eqnarray}
                A_{t+1} & = & \\frac{(1+q)A_t+2\\left(1+\\sqrt{(1+A_t)(1+qA_t)}\\right)}{(1-q)^2},\\\\
                \\beta_{t+1} & = & \\frac{A_t}{(1-q)A_{t+1}},\\\\
                \\delta_{t+1} & = & \\frac{1}{2}\\frac{(1-q)^2A_{t+1}-(1+q)A_t}{1+q+q A_t},
            \\end{eqnarray}

    with :math:`A_0=0`.

    **Theoretical guarantee**:
    A tight worst-case guarantee can be found in [1, Theorem 3]:

    .. math:: \\|z_n - x_\\star\\|^2 \\leqslant \\frac{1}{1+q A_n} \\|z_0-x_\\star\\|^2,

    where tightness is obtained on some quadratic loss functions (see [1, Lemma 2]).

    **References**:

    `[1] A. Taylor, Y. Drori (2022).
    An optimal gradient method for smooth strongly convex minimization.
    Mathematical Programming.
    <https://arxiv.org/pdf/2101.09741.pdf>`_

    Args:
        mu (float): the strong convexity parameter.
        L (float): the smoothness parameter.
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
        >>> pepit_tau, theoretical_tau = wc_information_theoretic(mu=.001, L=1, n=15, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 17x17
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 240 scalar constraint(s) ...
        			Function 1 : 240 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.7566088333863785
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite
        		All the primal scalar constraints are verified up to an error of 2.6490565304427935e-09
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 5.0900580550698854e-05
        (PEPit) Final upper bound (dual): 0.7566088219545419 and lower bound (primal example): 0.7566088333863785 
        (PEPit) Duality gap: absolute: -1.1431836588471356e-08 and relative: -1.5109308911059783e-08
        *** Example file: worst-case performance of the information theoretic exact method ***
        	PEP-it guarantee:	 ||z_n - x_* ||^2 <= 0.756609 ||z_0 - x_*||^2
        	Theoretical guarantee:	 ||z_n - x_* ||^2 <= 0.756605 ||z_0 - x_*||^2
    
    """
    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting point z0 of the algorithm
    z0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between z0 and x^*
    problem.set_initial_condition((z0 - xs) ** 2 <= 1)

    # Run n steps of the information theoretic exact method
    A_new = 0
    q = mu / L

    x = z0
    z = z0

    for i in range(n):
        A_old = A_new
        A_new = ((1 + q) * A_old + 2 * (1 + sqrt((1 + A_old) * (1 + q * A_old)))) / (1 - q) ** 2
        beta = A_old / (1 - q) / A_new
        delta = 1 / 2 * ((1 - q) ** 2 * A_new - (1 + q) * A_old) / (1 + q + q * A_old)

        y = (1 - beta) * z + beta * x
        x = y - 1 / L * func.gradient(y)
        z = (1 - q * delta) * z + q * delta * y - delta / L * func.gradient(y)

    # Set the performance metric to the distance accuracy
    problem.set_performance_metric((z - xs) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Theoretical guarantee (for comparison)
    theoretical_tau = 1 / (1 + q * A_new)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the information theoretic exact method ***')
        print('\tPEP-it guarantee:\t ||z_n - x_* ||^2 <= {:.6} ||z_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||z_n - x_* ||^2 <= {:.6} ||z_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_information_theoretic(mu=.001, L=1, n=15, wrapper="cvxpy", solver=None, verbose=1)
