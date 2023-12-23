from PEPit import PEP
from PEPit.functions import ConvexIndicatorFunction
from PEPit.functions import SmoothConvexFunction
from PEPit.primitive_steps import linear_optimization_step


def wc_frank_wolfe(L, D, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the composite convex minimization problem

    .. math:: F_\\star \\triangleq \\min_x \\{F(x) \\equiv f_1(x) + f_2(x)\\},

    where :math:`f_1` is :math:`L`-smooth and convex
    and where :math:`f_2` is a convex indicator function on :math:`\\mathcal{D}` of diameter at most :math:`D`.

    This code computes a worst-case guarantee for the **conditional gradient** method, aka **Frank-Wolfe** method,
    and looks for a low-dimensional worst-case example nearly achieving this worst-case guarantee using
    :math:`12` iterations of the logdet heuristic.
    That is, it computes the smallest possible :math:`\\tau(n, L)` such that the guarantee

    .. math :: F(x_n) - F(x_\\star) \\leqslant \\tau(n, L) D^2,

    is valid, where x_n is the output of the **conditional gradient** method,
    and where :math:`x_\\star` is a minimizer of :math:`F`.
    In short, for given values of :math:`n` and :math:`L`, :math:`\\tau(n, L)` is computed as the worst-case value of
    :math:`F(x_n) - F(x_\\star)` when :math:`D \\leqslant 1`. Then, it looks for a low-dimensional nearly achieving this
    performance.

    **Algorithm**:

    This method was first presented in [1]. A more recent version can be found in, e.g., [2, Algorithm 1].
    For :math:`t \\in \\{0, \\dots, n-1\\}`,

        .. math::
            \\begin{eqnarray}
                y_t & = & \\arg\\min_{s \\in \\mathcal{D}} \\langle s \\mid \\nabla f_1(x_t) \\rangle, \\\\
                x_{t+1} & = & \\frac{t}{t + 2} x_t + \\frac{2}{t + 2} y_t.
            \\end{eqnarray}

    **Theoretical guarantee**:

    An **upper** guarantee obtained in [2, Theorem 1] is

        .. math :: F(x_n) - F(x_\\star) \\leqslant \\frac{2L D^2}{n+2}.

    **References**: The algorithm is presented in, among others, [1, 2]. The logdet heuristic is presented in [3].

    [1] M .Frank, P. Wolfe (1956).
    An algorithm for quadratic programming.
    Naval research logistics quarterly, 3(1-2), 95-110.

    `[2] M. Jaggi (2013). Revisiting Frank-Wolfe: Projection-free sparse convex optimization.
    In 30th International Conference on Machine Learning (ICML).
    <http://proceedings.mlr.press/v28/jaggi13.pdf>`_

    `[3] F. Maryam, H. Hindi, S. Boyd (2003). Log-det heuristic for matrix rank minimization with applications to Hankel
    and Euclidean distance matrices. American Control Conference (ACC).
    <https://web.stanford.edu/~boyd/papers/pdf/rank_min_heur_hankel.pdf>`_

    Args:
        L (float): the smoothness parameter.
        D (float): diameter of :math:`f_2`.
        n (int): number of iterations.
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + solver details.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (float): theoretical value.

    Example:
        >>> pepit_tau, theoretical_tau = wc_frank_wolfe(L=1, D=1, n=10, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 26x26
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (0 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
        			Function 1 : Adding 132 scalar constraint(s) ...
        			Function 1 : 132 scalar constraint(s) added
        			Function 2 : Adding 325 scalar constraint(s) ...
        			Function 2 : 325 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.07828953904645822
        (PEPit) Postprocessing: 15 eigenvalue(s) > 1.6888130186731306e-08 before dimension reduction
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.07818953725941896
        (PEPit) Postprocessing: 11 eigenvalue(s) > 1.1861651266980264e-08 after 1 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.07818953903506054
        (PEPit) Postprocessing: 11 eigenvalue(s) > 0 after 2 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.07818953903905457
        (PEPit) Postprocessing: 11 eigenvalue(s) > 0 after 3 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.07818953893599122
        (PEPit) Postprocessing: 11 eigenvalue(s) > 0 after 4 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.07818953897127347
        (PEPit) Postprocessing: 11 eigenvalue(s) > 0 after 5 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.07818953853593198
        (PEPit) Postprocessing: 11 eigenvalue(s) > 0 after 6 dimension reduction step(s)
        (PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.07818953853593198
        (PEPit) Postprocessing: 11 eigenvalue(s) > 0 after dimension reduction
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 4.838754901390161e-10
        		All the primal scalar constraints are verified up to an error of 9.411701695771768e-10
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 3.474580080029191e-09
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 1.2084335345375351e-07
        (PEPit) Final upper bound (dual): 0.07828954284798424 and lower bound (primal example): 0.07818953853593198 
        (PEPit) Duality gap: absolute: 0.00010000431205225979 and relative: 0.0012789986221277267
        *** Example file: worst-case performance of the Conditional Gradient (Frank-Wolfe) in function value ***
        	PEPit guarantee:	 f(x_n)-f_* == 0.0782895 ||x0 - xs||^2
        	Theoretical guarantee:	 f(x_n)-f_* <= 0.166667 ||x0 - xs||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function and a convex indicator of rayon D
    func1 = problem.declare_function(function_class=SmoothConvexFunction, L=L)
    func2 = problem.declare_function(function_class=ConvexIndicatorFunction, D=D)
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Enforce the feasibility of x0 : there is no initial constraint on x0
    _ = func1.value(x0)
    _ = func2.value(x0)

    # Compute n steps of the Conditional Gradient / Frank-Wolfe method starting from x0
    x = x0
    for i in range(n):
        g = func1.gradient(x)
        y, _, _ = linear_optimization_step(g, func2)
        lam = 2 / (i + 2)
        x = (1 - lam) * x + lam * y

    # Set the performance metric to the final distance in function values to optimum
    problem.set_performance_metric((func.value(x)) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose,
                              dimension_reduction_heuristic="logdet6")

    # Compute theoretical guarantee (for comparison)
    # when theta = 1
    theoretical_tau = 2 * L * D ** 2 / (n + 2)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file:'
              ' worst-case performance of the Conditional Gradient (Frank-Wolfe) in function value ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* == {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(theoretical_tau))
    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_frank_wolfe(L=1, D=1, n=10, wrapper="cvxpy", solver=None, verbose=1)
