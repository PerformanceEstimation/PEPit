from PEPit import PEP
from PEPit.functions import ConvexQGFunction


def wc_heavy_ball_momentum_qg_convex(L, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is quadratically upper bounded (:math:`\\text{QG}^+` [2]), i.e.
    :math:`\\forall x, f(x) - f_\\star \\leqslant \\frac{L}{2} \\|x-x_\\star\\|^2`, and convex.

    This code computes a worst-case guarantee for the **Heavy-ball (HB)** method, aka **Polyak momentum** method.
    That is, it computes the smallest possible :math:`\\tau(n, L)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the **Heavy-ball (HB)** method,
    and where :math:`x_\\star` is the minimizer of :math:`f`.
    In short, for given values of :math:`n` and :math:`L`,
    :math:`\\tau(n, L)` is computed as the worst-case value of
    :math:`f(x_n)-f_\\star` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:

    This method is described in [1]

        .. math:: x_{t+1} = x_t - \\alpha_t \\nabla f(x_t) + \\beta_t (x_t-x_{t-1})

        with

        .. math:: \\alpha_t = \\frac{1}{L} \\frac{1}{t+2}

        and

        .. math:: \\beta_t = \\frac{t}{t+2}

    **Theoretical guarantee**:

    The **tight** guarantee obtained in [2, Theorem 2.3] (lower) and [2, Theorem 2.4] (upper) is

        .. math:: f(x_n) - f_\\star \\leqslant \\frac{L}{2}\\frac{1}{n+1} \\|x_0 - x_\\star\\|^2.

    **References**:
    This methods was first introduce in [1, section 3],
    and convergence **tight** bound was proven in [2, Theorem 2.3] (lower) and [2, Theorem 2.4] (upper).

    `[1] E. Ghadimi, H. R. Feyzmahdavian, M. Johansson (2015).
    Global convergence of the Heavy-ball method for convex optimization.
    European Control Conference (ECC).
    <https://arxiv.org/pdf/1412.7457.pdf>`_

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
        >>> pepit_tau, theoretical_tau = wc_heavy_ball_momentum_qg_convex(L=1, n=5, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 8x8
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 48 scalar constraint(s) ...
        			Function 1 : 48 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.08333333312648067
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite
        		All the primal scalar constraints are verified up to an error of 1.0360445487633818e-10
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 9.384283773139436e-11
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 2.3121924686696765e-09
        (PEPit) Final upper bound (dual): 0.08333333333812117 and lower bound (primal example): 0.08333333312648067 
        (PEPit) Duality gap: absolute: 2.1164049679445185e-10 and relative: 2.539685967837512e-09
        *** Example file: worst-case performance of the Heavy-Ball method ***
        	PEPit guarantee:	 f(x_n)-f_* <= 0.0833333 ||x_0 - x_*||^2
        	Theoretical guarantee:	 f(x_n)-f_* <= 0.0833333 ||x_0 - x_*||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(ConvexQGFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding function value f0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between f(x0) and f(x^*)
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run one step of the heavy ball method
    x_new = x0
    x_old = x0

    for t in range(n):
        x_next = x_new - 1 / (L * (t + 2)) * func.gradient(x_new) + t / (t + 2) * (x_new - x_old)
        x_old = x_new
        x_new = x_next

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric(func.value(x_new) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / (2 * (n + 1))

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the Heavy-Ball method ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_heavy_ball_momentum_qg_convex(L=1, n=5, wrapper="cvxpy", solver=None, verbose=1)
