from PEPit import PEP
from PEPit.functions import ConvexFunction
from PEPit.functions import SmoothConvexFunction
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.primitive_steps import proximal_step


def wc_three_operator_splitting(mu1, L1, L3, alpha, theta, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the composite convex minimization problem,

    .. math:: F_\\star \\triangleq \\min_x \\{F(x) \\equiv f_1(x) + f_2(x) + f_3(x)\\}

    where, :math:`f_1` is :math:`L_1`-smooth and :math:`\\mu_1`-strongly convex,
    :math:`f_2` is closed, convex and proper,
    and :math:`f_3` is :math:`L_3`-smooth convex.
    Proximal operators are assumed to be available for :math:`f_1` and :math:`f_2`.

    This code computes a worst-case guarantee for the **Three Operator Splitting (TOS)**.
    That is, it computes the smallest possible :math:`\\tau(n, L_1, L_3, \\mu_1)` such that the guarantee

    .. math:: \\|w^{(0)}_{n} - w^{(1)}_{n}\\|^2 \\leqslant \\tau(n, L_1, L_3, \\mu_1, \\alpha, \\theta) \\|w^{(0)}_{0} - w^{(1)}_{0}\\|^2

    is valid, where :math:`w^{(0)}_{0}` and :math:`w^{(1)}_{0}` are two different starting points
    and :math:`w^{(0)}_{n}` and :math:`w^{(1)}_{n}` are the two corresponding :math:`n^{\\mathrm{th}}` outputs of TOS.
    (i.e., how do the iterates contract when the method is started from two different initial points).

    In short, for given values of :math:`n`, :math:`L_1`, :math:`L_3`, :math:`\\mu_1`, :math:`\\alpha`
    and :math:`\\theta`, the contraction factor :math:`\\tau(n, L_1, L_3, \\mu_1, \\alpha, \\theta)`
    is computed as the worst-case value of :math:`\\|w^{(0)}_{n} - w^{(1)}_{n}\\|^2`
    when :math:`\\|w^{(0)}_{0} - w^{(1)}_{0}\\|^2 \\leqslant 1`.

    **Algorithm**:
    One iteration of the algorithm is described in [1]. For :math:`t \\in \\{0, \\dots, n-1\\}`,

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_t & = & \\mathrm{prox}_{\\alpha, f_2}(w_t), \\\\
                y_t & = & \\mathrm{prox}_{\\alpha, f_1}(2 x_t - w_t - \\alpha \\nabla f_3(x_t)), \\\\
                w_{t+1} & = & w_t + \\theta (y_t - x_t).
            \\end{eqnarray}

    **References**: The TOS was introduced in [1].

    `[1] D. Davis, W. Yin (2017).
    A three-operator splitting scheme and its optimization applications.
    Set-valued and variational analysis, 25(4), 829-858.
    <https://arxiv.org/pdf/1504.01032.pdf>`_

    Args:
        mu1 (float): the strong convexity parameter of function :math:`f_1`.
        L1 (float): the smoothness parameter of function :math:`f_1`.
        L3 (float): the smoothness parameter of function :math:`f_3`.
        alpha (float): parameter of the scheme.
        theta (float): parameter of the scheme.
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
        theoretical_tau (None): no theoretical value.

    Example:
        >>> L3 = 1
        >>> alpha = 1 / L3
        >>> pepit_tau, theoretical_tau = wc_three_operator_splitting(mu1=0.1, L1=10, L3=L3, alpha=alpha, theta=1, n=4, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 26x26
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 3 function(s)
        			Function 1 : Adding 56 scalar constraint(s) ...
        			Function 1 : 56 scalar constraint(s) added
        			Function 2 : Adding 56 scalar constraint(s) ...
        			Function 2 : 56 scalar constraint(s) added
        			Function 3 : Adding 56 scalar constraint(s) ...
        			Function 3 : 56 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.4754523280192519
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 6.32207763279334e-10
        		All the primal scalar constraints are verified up to an error of 2.2438998784068964e-09
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 5.155823636112884e-10
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 5.293077507135036e-07
        (PEPit) Final upper bound (dual): 0.475452328074928 and lower bound (primal example): 0.4754523280192519 
        (PEPit) Duality gap: absolute: 5.5676074861565894e-11 and relative: 1.1710127720588522e-10
        *** Example file: worst-case performance of the Three Operator Splitting in distance ***
        	PEPit guarantee:	 ||w^2_n - w^1_n||^2 <= 0.475452 ||x0 - ws||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function, a smooth function and a convex function.
    func1 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu1, L=L1)
    func2 = problem.declare_function(ConvexFunction)
    func3 = problem.declare_function(SmoothConvexFunction, L=L3)

    # Then define the starting points w0 and w0p of the algorithm
    w0 = problem.set_initial_point()
    w0p = problem.set_initial_point()

    # Set the initial constraint that is the distance between w0 and w0p
    problem.set_initial_condition((w0 - w0p) ** 2 <= 1)

    # Compute n steps of the Three Operator Splitting starting from w0
    w = w0
    for _ in range(n):
        x, _, _ = proximal_step(w, func2, alpha)
        gx, _ = func3.oracle(x)
        y, _, _ = proximal_step(2 * x - w - alpha * gx, func1, alpha)
        w = w + theta * (y - x)

    # Compute trajectory starting from w0p
    wp = w0p
    for _ in range(n):
        xp, _, _ = proximal_step(wp, func2, alpha)
        gxp, _ = func3.oracle(xp)
        yp, _, _ = proximal_step(2 * xp - wp - alpha * gxp, func1, alpha)
        wp = wp + theta * (yp - xp)

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric((w - wp) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = None

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the Three Operator Splitting in distance ***')
        print('\tPEPit guarantee:\t ||w^2_n - w^1_n||^2 <= {:.6} ||x0 - ws||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L3 = 1
    alpha = 1 / L3
    pepit_tau, theoretical_tau = wc_three_operator_splitting(mu1=0.1, L1=10, L3=L3, alpha=alpha, theta=1, n=4,
                                                             wrapper="cvxpy", solver=None,
                                                             verbose=1)
