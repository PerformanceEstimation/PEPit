from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.primitive_steps import inexact_gradient_step


def wc_inexact_gradient(L, mu, epsilon, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for an **inexact gradient method** and looks for a low-dimensional
    worst-case example nearly achieving this worst-case guarantee using :math:`10` iterations of the logdet heuristic.
    
    That is, it computes the smallest possible :math:`\\tau(n,L,\\mu,\\varepsilon)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n,L,\\mu,\\varepsilon) (f(x_0) - f_\\star)

    is valid, where :math:`x_n` is the output of the gradient descent with an inexact descent direction,
    and where :math:`x_\\star` is the minimizer of :math:`f`. Then, it looks for a low-dimensional nearly achieving this
    performance.

    The inexact descent direction is assumed to satisfy a relative inaccuracy
    described by (with :math:`0 \\leqslant \\varepsilon \\leqslant 1`)

    .. math:: \|\\nabla f(x_t) - d_t\| \\leqslant \\varepsilon \\|\\nabla f(x_t)\\|,

    where :math:`\\nabla f(x_t)` is the true gradient,
    and :math:`d_t` is the approximate descent direction that is used.

    **Algorithm**:

    The inexact gradient descent under consideration can be written as

        .. math:: x_{t+1} = x_t - \\frac{2}{L_{\\varepsilon} + \\mu_{\\varepsilon}} d_t

    where :math:`d_t` is the inexact search direction, :math:`L_{\\varepsilon} = (1 + \\varepsilon)L`
    and :math:`\mu_{\\varepsilon} = (1-\\varepsilon) \\mu`.

    **Theoretical guarantee**:

    A **tight** worst-case guarantee obtained in [1, Theorem 5.3] or [2, Remark 1.6] is

        .. math:: f(x_n) - f_\\star \\leqslant \\left(\\frac{L_{\\varepsilon} - \\mu_{\\varepsilon}}{L_{\\varepsilon} + \\mu_{\\varepsilon}}\\right)^{2n}(f(x_0) - f_\\star ),

    with :math:`L_{\\varepsilon} = (1 + \\varepsilon)L` and :math:`\mu_{\\varepsilon} = (1-\\varepsilon) \\mu`. This
    guarantee is achieved on one-dimensional quadratic functions.

    **References**:The detailed analyses can be found in [1, 2]. The logdet heuristic is presented in [3].

    `[1] E. De Klerk, F. Glineur, A. Taylor (2020). Worst-case convergence analysis of
    inexact gradient and Newton methods through semidefinite programming performance estimation.
    SIAM Journal on Optimization, 30(3), 2053-2082.
    <https://arxiv.org/pdf/1709.05191.pdf>`_

    `[2] O. Gannot (2021). A frequency-domain analysis of inexact gradient methods.
    Mathematical Programming.
    <https://arxiv.org/pdf/1912.13494.pdf>`_

    `[3] F. Maryam, H. Hindi, S. Boyd (2003). Log-det heuristic for matrix rank minimization with applications to Hankel
    and Euclidean distance matrices. American Control Conference (ACC).
    <https://web.stanford.edu/~boyd/papers/pdf/rank_min_heur_hankel.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        epsilon (float): level of inaccuracy
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
        >>> pepit_tau, theoretical_tau = wc_inexact_gradient(L=1, mu=0.1, epsilon=0.1, n=6, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 15x15
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 56 scalar constraint(s) ...
        			Function 1 : 56 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 1 function(s)
        			Function 1 : Adding 6 scalar constraint(s) ...
        			Function 1 : 6 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.13973101296945833
        (PEPit) Postprocessing: 3 eigenvalue(s) > 2.99936924706653e-06 before dimension reduction
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.13963101314530774
        (PEPit) Postprocessing: 1 eigenvalue(s) > 7.356394561198437e-08 after 1 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.13963101333230654
        (PEPit) Postprocessing: 1 eigenvalue(s) > 0 after 2 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.13963101314132523
        (PEPit) Postprocessing: 1 eigenvalue(s) > 0 after 3 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.13963101315746582
        (PEPit) Postprocessing: 1 eigenvalue(s) > 0 after 4 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.13963101311324322
        (PEPit) Postprocessing: 1 eigenvalue(s) > 0 after 5 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.13963101297326166
        (PEPit) Postprocessing: 1 eigenvalue(s) > 0 after 6 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.13963101303966996
        (PEPit) Postprocessing: 1 eigenvalue(s) > 1.216071300777247e-12 after 7 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.13963101306836934
        (PEPit) Postprocessing: 1 eigenvalue(s) > 7.45808567822026e-12 after 8 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.13963101314164972
        (PEPit) Postprocessing: 1 eigenvalue(s) > 6.382593300104802e-12 after 9 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.13963101344891796
        (PEPit) Postprocessing: 1 eigenvalue(s) > 0 after 10 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.13963101344891796
        (PEPit) Postprocessing: 1 eigenvalue(s) > 0 after dimension reduction
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 3.749538388559788e-11
        		All the primal scalar constraints are verified up to an error of 6.561007293015564e-11
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 3.481776410060961e-07
        (PEPit) Final upper bound (dual): 0.13973101221655027 and lower bound (primal example): 0.13963101344891796 
        (PEPit) Duality gap: absolute: 9.999876763230886e-05 and relative: 0.0007161644477277393
        *** Example file: worst-case performance of inexact gradient ***
        	PEPit guarantee:	 f(x_n)-f_* == 0.139731 (f(x_0)-f_*)
        	Theoretical guarantee:	 f(x_n)-f_* <= 0.139731 (f(x_0)-f_*)
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    # as well as corresponding inexact gradient and function value g0 and f0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between f0 and f_*
    problem.set_initial_condition(func(x0) - fs <= 1)

    # Run n steps of the inexact gradient method
    Leps = (1 + epsilon) * L
    meps = (1 - epsilon) * mu
    gamma = 2 / (Leps + meps)

    x = x0
    for i in range(n):
        x, dx, fx = inexact_gradient_step(x, func, gamma=gamma, epsilon=epsilon, notion='relative')

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func(x) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose,
                              dimension_reduction_heuristic="logdet10")

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = ((Leps - meps) / (Leps + meps)) ** (2 * n)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of inexact gradient ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* == {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_inexact_gradient(L=1, mu=0.1, epsilon=0.1, n=6,
                                                     wrapper="cvxpy", solver=None,
                                                     verbose=1)
