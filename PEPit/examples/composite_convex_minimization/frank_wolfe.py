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

    This code computes a worst-case guarantee for the **conditional gradient** method, aka **Frank-Wolfe** method.
    That is, it computes the smallest possible :math:`\\tau(n, L)` such that the guarantee

    .. math :: F(x_n) - F(x_\\star) \\leqslant \\tau(n, L) D^2,

    is valid, where x_n is the output of the **conditional gradient** method,
    and where :math:`x_\\star` is a minimizer of :math:`F`.
    In short, for given values of :math:`n` and :math:`L`, :math:`\\tau(n, L)` is computed as the worst-case value of
    :math:`F(x_n) - F(x_\\star)` when :math:`D \\leqslant 1`.

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

    **References**:

    [1] M .Frank, P. Wolfe (1956).
    An algorithm for quadratic programming.
    Naval research logistics quarterly, 3(1-2), 95-110.

    `[2] M. Jaggi (2013).
    Revisiting Frank-Wolfe: Projection-free sparse convex optimization.
    In 30th International Conference on Machine Learning (ICML).
    <http://proceedings.mlr.press/v28/jaggi13.pdf>`_

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
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 4.140365475126263e-09
        		All the primal scalar constraints are verified up to an error of 7.758491793463662e-09
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 3.474580080029191e-09
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 1.2084335345375351e-07
        (PEPit) Final upper bound (dual): 0.07828954284798424 and lower bound (primal example): 0.07828953904645822 
        (PEPit) Duality gap: absolute: 3.801526024527213e-09 and relative: 4.855726666459652e-08
        *** Example file: worst-case performance of the Conditional Gradient (Frank-Wolfe) in function value ***
        	PEPit guarantee:	 f(x_n)-f_* <= 0.0782895 ||x0 - xs||^2
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
    fs = func(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Enforce the feasibility of x0 : there is no initial constraint on x0
    _ = func1(x0)
    _ = func2(x0)

    # Compute n steps of the Conditional Gradient / Frank-Wolfe method starting from x0
    x = x0
    for i in range(n):
        g = func1.gradient(x)
        y, _, _ = linear_optimization_step(g, func2)
        lam = 2 / (i + 2)
        x = (1 - lam) * x + lam * y

    # Set the performance metric to the final distance in function values to optimum
    problem.set_performance_metric((func(x)) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    # when theta = 1
    theoretical_tau = 2 * L * D ** 2 / (n + 2)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file:'
              ' worst-case performance of the Conditional Gradient (Frank-Wolfe) in function value ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(theoretical_tau))
    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_frank_wolfe(L=1, D=1, n=10, wrapper="cvxpy", solver=None, verbose=1)
