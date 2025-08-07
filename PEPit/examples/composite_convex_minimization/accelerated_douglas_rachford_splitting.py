from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.functions import ConvexFunction
from PEPit.primitive_steps import proximal_step


def wc_accelerated_douglas_rachford_splitting(mu, L, alpha, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the composite convex minimization problem

    .. math:: F_\\star \\triangleq \\min_x \\{F(x) \\equiv f_1(x) + f_2(x)\\},

    where :math:`f_1` is closed convex and proper, and :math:`f_2` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for **accelerated Douglas-Rachford**. That is, it computes
    the smallest possible :math:`\\tau(n, L, \\mu, \\alpha)` such that the guarantee

    .. math:: F(y_n) - F(x_\\star) \\leqslant \\tau(n,L,\\mu,\\alpha) \\|w_0 - w_\\star\\|^2

    is valid, :math:`\\alpha` is a parameter of the method, and where :math:`y_n` is the output
    of the accelerated Douglas-Rachford Splitting method, where :math:`x_\\star` is a minimizer of :math:`F`,
    and :math:`w_\\star` defined such that

    .. math:: x_\\star = \\mathrm{prox}_{\\alpha f_2}(w_\\star)

    is an optimal point.

    In short, for given values of :math:`n`, :math:`L`, :math:`\\mu`, :math:`\\alpha`,
    :math:`\\tau(n, L, \\mu, \\alpha)` is computed as the worst-case value of :math:`F(y_n)-F_\\star`
    when :math:`\|w_0 - w_\\star\|^2 \\leqslant 1`.

    **Algorithm**:
    The accelerated Douglas-Rachford splitting is described in [1, Section 4]. For :math:`t \\in \\{0, \\dots, n-1\\}`,

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_{t} & = & \\mathrm{prox}_{\\alpha f_2} (u_t),\\\\
                y_{t} & = & \\mathrm{prox}_{\\alpha f_1}(2x_t-u_t),\\\\
                w_{t+1} & = & u_t + \\theta (y_t-x_t),\\\\
                u_{t+1} & = & \\left\\{\\begin{array}{ll} w_{t+1}+\\frac{t-1}{t+2}(w_{t+1}-w_t)\, & \\text{if } t >1,\\\\
                w_{t+1} & \\text{otherwise.} \\end{array}\\right.
            \\end{eqnarray}

    **Theoretical guarantee**:
    There is no known worst-case guarantee for this method beyond quadratic minimization.
    For quadratics, an **upper** bound on is provided by [1, Theorem 5]:

    .. math:: F(y_n) - F_\\star \\leqslant \\frac{2}{\\alpha \\theta (n + 3)^ 2} \|w_0-w_\\star\|^2,

    when :math:`\\theta=\\frac{1-\\alpha L}{1+\\alpha L}` and :math:`\\alpha < \\frac{1}{L}`.

    **References**:
    An analysis of the accelerated Douglas-Rachford splitting is available in [1, Theorem 5] for when the convex
    minimization problem is quadratic.

    `[1] P. Patrinos, L. Stella, A. Bemporad (2014).
    Douglas-Rachford splitting: Complexity estimates and accelerated variants.
    In 53rd IEEE Conference on Decision and Control (CDC).
    <https://arxiv.org/pdf/1407.6723.pdf>`_

    Args:
        mu (float): the strong convexity parameter.
        L (float): the smoothness parameter.
        alpha (float): the parameter of the scheme.
        n (int): the number of iterations.
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): level of information details to print.

                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + solver details.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (float): theoretical value (upper bound for quadratics; not directly comparable).

    Example:
        >>> pepit_tau, theoretical_tau = wc_accelerated_douglas_rachford_splitting(mu=.1, L=1, alpha=.9, n=2, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 9x9
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
        			Function 1 : Adding 12 scalar constraint(s) ...
        			Function 1 : 12 scalar constraint(s) added
        			Function 2 : Adding 12 scalar constraint(s) ...
        			Function 2 : 12 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.19291482276257793
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 6.6554381338528585e-09
        		All the primal scalar constraints are verified up to an error of 1.4452049501567643e-08
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 7.332184313444333e-08
        (PEPit) Final upper bound (dual): 0.19291482672193344 and lower bound (primal example): 0.19291482276257793 
        (PEPit) Duality gap: absolute: 3.959355510119167e-09 and relative: 2.0523853239582232e-08
        *** Example file: worst-case performance of the Accelerated Douglas Rachford Splitting in function values ***
        	PEPit guarantee:			 F(y_n)-F_* <= 0.192915 ||x0 - ws||^2
        	Theoretical guarantee for quadratics:	 F(y_n)-F_* <= 1.68889 ||x0 - ws||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function and a smooth strongly convex function
    func1 = problem.declare_function(ConvexFunction)
    func2 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    fs = func(xs)
    g1s, _ = func1.oracle(xs)
    g2s, _ = func2.oracle(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Set the parameters of the scheme
    theta = (1 - alpha * L) / (1 + alpha * L)

    # Set the initial constraint that is the distance between x0 and ws = w^*
    ws = xs + alpha * g2s
    problem.set_initial_condition((ws - x0) ** 2 <= 1)

    # Compute n steps of the Accelerated Douglas Rachford Splitting starting from x0
    x = [x0 for _ in range(n)]
    w = [x0 for _ in range(n + 1)]
    u = [x0 for _ in range(n + 1)]
    for i in range(n):
        x[i], _, _ = proximal_step(u[i], func2, alpha)
        y, _, fy = proximal_step(2 * x[i] - u[i], func1, alpha)
        w[i + 1] = u[i] + theta * (y - x[i])
        if i >= 1:
            u[i + 1] = w[i + 1] + (i - 1) / (i + 2) * (w[i + 1] - w[i])
        else:
            u[i + 1] = w[i + 1]

    # Set the performance metric to the final distance in function values to optimum
    problem.set_performance_metric(func2(y) + fy - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    if alpha < 1 / L:
        theoretical_tau = 2 / (alpha * theta * (n + 3) ** 2)
    else:
        theoretical_tau = None

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file:'
              ' worst-case performance of the Accelerated Douglas Rachford Splitting in function values ***')
        print('\tPEPit guarantee:\t\t\t F(y_n)-F_* <= {:.6} ||x0 - ws||^2'.format(pepit_tau))
        if alpha < 1 / L:
            print('\tTheoretical guarantee for quadratics:\t F(y_n)-F_* <= {:.6} ||x0 - ws||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_accelerated_douglas_rachford_splitting(mu=.1, L=1, alpha=.9, n=2,
                                                                           wrapper="cvxpy", solver=None,
                                                                           verbose=1)
