from PEPit import PEP
from PEPit.functions import ConvexFunction
from PEPit.primitive_steps import bregman_proximal_step


def wc_bregman_proximal_point(gamma, n, verbose=1):
    """
    Consider the composite convex minimization problem

        .. math:: F_\\star \\triangleq \min_x \\{F(x) \equiv f_1(x)+f_2(x) \\}

    where :math:`f_1(x)` and :math:`f_2(x)` are closed convex proper functions.

    This code computes a worst-case guarantee for **Bregman Proximal Point** method.
    That is, it computes the smallest possible :math:`\\tau(n, \\gamma)` such that the guarantee

        .. math:: F(x_n) - F(x_\\star) \\leqslant \\tau(n, \gamma) D_{f_1}(x_\\star; x_0)

    is valid, where :math:`x_n` is the output of the **Bregman Proximal Point** (BPP) method,
    where :math:`x_\\star` is a minimizer of :math:`F`, and when :math:`D_{f_1}` is the Bregman distance generated by :math:`f_1`.

    **Algorithm**: Bregman proximal point is described in [1, Section 2, equation (9)]. For :math:`t \\in \\{0, \\dots, n-1\\}`,

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_{t+1} & = & \\arg\\min_{u \\in R^n} f_1(u) + \\frac{1}{\\gamma} D_{f_2}(u; x_t), \\\\
                D_h(x; y) & = & h(x) - h(y) - \\nabla h (y)^T(x - y).
            \\end{eqnarray}

    **Theoretical guarantee**: A **tight** empirical guarantee can be guessed from the numerics

        .. math:: F(x_n) - F(x_\\star) \\leqslant \\frac{1}{\\gamma n} D_{f_1}(x_\\star, x_0).

    **References**:

    `[1] Y. Censor, S.A. Zenios (1992). Proximal minimization algorithm with D-functions.
    Journal of Optimization Theory and Applications, 73(3), 451-464.
    <https://link.springer.com/content/pdf/10.1007/BF00940051.pdf>`_

    Args:
        gamma (float): step-size.
        n (int): number of iterations.
        verbose (int): Level of information details to print.

                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (float): theoretical value.

    Examples:
        >>> pepit_tau, theoretical_tau = wc_bregman_proximal_point(gamma=3, n=5, verbose=1)
        (PEPit) Setting up the problem: size of the main PSD matrix: 14x14
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
                         function 1 : Adding 30 scalar constraint(s) ...
                         function 1 : 30 scalar constraint(s) added
                         function 2 : Adding 42 scalar constraint(s) ...
                         function 2 : 42 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.06666740784196148
        *** Example file: worst-case performance of the Bregman Proximal Point in function values ***
                PEPit guarantee:         F(x_n)-F_* <= 0.0666674 Dh(x_*; x_0)
                Theoretical guarantee:   F(x_n)-F_* <= 0.0666667 Dh(x_*; x_0)

    """

    # Instantiate PEP
    problem = PEP()

    # Declare three convex functions
    func1 = problem.declare_function(ConvexFunction)
    func2 = problem.declare_function(ConvexFunction)

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func1.stationary_point()
    fs = func1(xs)
    gf2s, f2s = func2.oracle(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()
    gf20, f20 = func2.oracle(x0)

    # Set the initial constraint that is the Bregman distance between x0 and x^*
    problem.set_initial_condition(f2s - f20 - gf20 * (xs - x0) <= 1)

    # Compute n steps of the Bregman Proximal Point method starting from x0
    gf2 = gf20
    for i in range(n):
        x, gf2, f2x, f1x, f1 = bregman_proximal_step(gf2, func2, func1, gamma)

    # Set the performance metric to the final distance in function values to optimum
    problem.set_performance_metric(f1 - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 1 / (gamma * n)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the Bregman Proximal Point in function values ***')
        print('\tPEPit guarantee:\t F(x_n)-F_* <= {:.6} Dh(x_*; x_0)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t F(x_n)-F_* <= {:.6} Dh(x_*; x_0)'.format(theoretical_tau))
    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_bregman_proximal_point(gamma=3, n=5, verbose=1)
