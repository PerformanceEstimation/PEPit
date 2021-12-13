from PEPit.pep import PEP
from PEPit.functions.smooth_convex_function import SmoothConvexFunction
from PEPit.functions.convex_function import ConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_douglas_rachford_splitting(L, alpha, theta, n, verbose=True):
    """
    Consider the composite convex minimization problem

        .. math:: F_\\star \\triangleq \min_x \\{F(x) \equiv f_1(x)+f_2(x) \\}

    where :math:`f_1(x)` is :math:`L`-smooth, and :math:`f_2` is convex,
    closed and proper. Both proximal operators are assumed to be available.

    This code computes a worst-case guarantee for the **Douglas Rachford Splitting (DRS)** method.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\alpha, \\theta)` such that the guarantee

        .. math:: F(y_n) - F(x_\\star) \\leqslant \\tau(n, L, \\alpha, \\theta) ||x_0 - x_\\star||^2.

    is valid, where it is known that :math:`x_k` and :math:`y_k` converge to :math:`x_\\star`, but not :math:`w_k`, and hence
    we require the initial condition on :math:`x_0`(arbitrary choice; partially justified by
    the fact we choose :math:`f_2` to be the smooth function).

    Note that :math:`y_N` is feasible as it
    has a finite value for :math:`f_1` (output of the proximal operator on :math:`f_1`) and as :math:`f_2` is smooth.

    **Algorithm**:

    Our notations for the DRS algorithm are as follows

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_t & = & \\mathrm{prox}_{\\alpha f_2}(w_t) \\\\
                y_t & = & \\mathrm{prox}_{\\alpha f_1}(2x_t - w_t) \\\\
                w_{t+1} & = & w_t + \\theta (y_t - x_t)
            \\end{eqnarray}

    **Theoretical guarantee**:

    TODO : find reference for theoretical upper bound?

        .. math:: F(y_n) - F(x_\\star)\\leqslant  \\frac{1}{n} ||x_0 - x_\\star||^2

    **References**:

    [1] Giselsson, Pontus, and Stephen Boyd. "Linear convergence and metric selection in
    Douglas-Rachford splitting and ADMM.

    Args:
        L (float): the smoothness parameter.
        alpha (float): parameter of the scheme.
        theta (float): parameter of the scheme.
        n (int): number of iterations.
        verbose (bool, optional): if True, print conclusion.

    Returns:
        tuple: worst-case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_douglas_rachford_splitting(L=1, alpha=1, theta=1, n=10, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 26x26
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 2 function(s)
                 function 1 : 132 constraint(s) added
                 function 2 : 156 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.02501210513106952
        *** Example file: worst-case performance of the Douglas Rachford Splitting in function values ***
            PEP-it guarantee:		 f(y_n)-f_* <= 0.0250121 ||x0 - xs||^2
            Theoretical guarantee :	 f(y_n)-f_* <= 0.0909091 ||x0 - xs||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex and a smooth strongly convex function.
    func1 = problem.declare_function(ConvexFunction, param={})
    func2 = problem.declare_function(SmoothConvexFunction, param={'L': L})
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()
    _ = func.value(x0)

    # Compute n steps of the Douglas Rachford Splitting starting from x0
    x = [x0 for _ in range(n)]
    w = [x0 for _ in range(n + 1)]
    for i in range(n):
        x[i], _, _ = proximal_step(w[i], func2, alpha)
        y, _, fy = proximal_step(2 * x[i] - w[i], func1, alpha)
        w[i + 1] = w[i] + theta * (y - x[i])

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x[0] - xs) ** 2 <= 1)

    # Set the performance metric to the final distance to the optimum in function values
    problem.set_performance_metric((func2.value(y) + fy) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    # when theta = 1
    theoretical_tau = 1 / (n + 1)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Douglas Rachford Splitting in function values ***')
        print('\tPEP-it guarantee:\t\t f(y_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee :\t f(y_n)-f_* <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))
    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_douglas_rachford_splitting(L=1, alpha=1, theta=1, n=10, verbose=True)
