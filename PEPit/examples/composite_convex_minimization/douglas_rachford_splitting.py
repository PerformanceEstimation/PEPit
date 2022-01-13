from PEPit import PEP
from PEPit.functions import SmoothConvexFunction
from PEPit.functions import ConvexFunction
from PEPit.primitive_steps import proximal_step


def wc_douglas_rachford_splitting(L, alpha, theta, n, verbose=True):
    """
    Consider the composite convex minimization problem

        .. math:: F_\\star \\triangleq \min_x \\{F(x) \equiv f_1(x)+f_2(x) \\}

    where :math:`f_1(x)` is is convex, closed and proper , and :math:`f_2` is :math:`L`-smooth.
    Both proximal operators are assumed to be available.

    This code computes a worst-case guarantee for the **Douglas Rachford Splitting (DRS)** method.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\alpha, \\theta)` such that the guarantee

        .. math:: F(y_n) - F(x_\\star) \\leqslant \\tau(n, L, \\alpha, \\theta) \\|x_0 - x_\\star\\|^2.

    is valid, where it is known that :math:`x_k` and :math:`y_k` converge to :math:`x_\\star`, but not :math:`w_k` (see definitions in the section **Algorithm**). Hence
    we require the initial condition on :math:`x_0` (arbitrary choice, partially justified by
    the fact we choose :math:`f_2` to be the smooth function).

    Note that :math:`y_n` is feasible as it
    has a finite value for :math:`f_1` (output of the proximal operator on :math:`f_1`) and as :math:`f_2` is smooth.

    **Algorithm**:

    Our notations for the DRS method are as follows, for :math:`t \\in \\{0, \\dots, n-1\\}`,

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_t & = & \\mathrm{prox}_{\\alpha f_2}(w_t), \\\\
                y_t & = & \\mathrm{prox}_{\\alpha f_1}(2x_t - w_t), \\\\
                w_{t+1} & = & w_t + \\theta (y_t - x_t).
            \\end{eqnarray}

    This description can be found in [1, Section 7.3].

    **Theoretical guarantee**: We compare the output with that of PESTO [2] for when :math:`0\\leqslant n \\leqslant 10`
    in the case where :math:`\\alpha=\\theta=L=1`.

    **References**:

    `[1] E. Ryu, S. Boyd (2016). A primer on monotone operator methods.
    Applied and Computational Mathematics 15(1), 3-43.
    <https://web.stanford.edu/~boyd/papers/pdf/monotone_primer.pdf>`_

    `[2] A. Taylor, J. Hendrickx, F. Glineur (2017). Performance Estimation Toolbox (PESTO): automated worst-case
    analysis of first-order optimization methods. In 56th IEEE Conference on Decision and Control (CDC).
    <https://github.com/AdrienTaylor/Performance-Estimation-Toolbox>`_

    Args:
        L (float): the smoothness parameter.
        alpha (float): parameter of the scheme.
        theta (float): parameter of the scheme.
        n (int): number of iterations.
        verbose (bool, optional): if True, print conclusion.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (float): theoretical value.

    Example:
        >>> pepit_tau, theoretical_tau = wc_douglas_rachford_splitting(L=1, alpha=1, theta=1, n=9, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 22x22
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
		         function 1 : 90 constraint(s) added
		         function 2 : 110 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); optimal value: 0.027791729882647262
        *** Example file: worst-case performance of the Douglas Rachford Splitting in function values ***
	        PEPit guarantee:       f(y_n)-f_* <= 0.0278 ||x0 - xs||^2
	        Theoretical guarantee:  f(y_n)-f_* <= 0.0278 ||x0 - xs||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex and a smooth convex function.
    func1 = problem.declare_function(ConvexFunction, param={})
    func2 = problem.declare_function(SmoothConvexFunction, param={'L': L})
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Compute n steps of the Douglas-Rachford splitting starting from x0
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

    # Compute theoretical guarantee (for comparison) when theta = 1
    if theta == 1 and alpha == 1 and L == 1 and 0 < n <= 10:
        pesto_tau = [1/4, 0.1273, 0.0838, 0.0627, 0.0501, 0.0417, 0.0357, 0.0313, 0.0278, 0.0250]
        theoretical_tau = pesto_tau[n-1]
    else:
        theoretical_tau = None

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Douglas Rachford Splitting in function values ***')
        print('\tPEPit guarantee:\t f(y_n)-f_* <= {:.3} ||x0 - xs||^2'.format(pepit_tau))
        if theta == 1 and alpha == 1 and L == 1 and n <= 10:
            print('\tTheoretical guarantee :\t f(y_n)-f_* <= {:.3} ||x0 - xs||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_douglas_rachford_splitting(L=1, alpha=1, theta=1, n=9, verbose=True)
