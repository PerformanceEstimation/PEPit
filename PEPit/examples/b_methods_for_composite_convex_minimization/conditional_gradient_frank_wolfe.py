from PEPit.pep import PEP
from PEPit.functions.smooth_convex_function import SmoothConvexFunction
from PEPit.functions.convex_indicator import ConvexIndicatorFunction
from PEPit.primitive_steps.linear_optimization_step import linear_optimization_step


def wc_cg_fw(L, D, n, verbose=True):
    """
    Consider the composite convex minimization problem

    .. math:: f_\star = \\min_x {F(x) = f_1(x) + f_2(x)},

    where :math:`f_1` is :math:`L`-smooth and convex
    and where :math:`f_2` is a convex indicator function on :math:`\\mathcal{D}` of diameter at most :math:`D`.

    This code computes a worst-case guarantee for the **conditional gradient** method.
    That is, it computes the smallest possible :math:`\\tau(n, L, D)` such that the guarantee

    .. math :: F(x_n) - F(x_\star) \\leqslant \\tau(n, L, D) \\|x_0 - x_\star\\|^2,

    is valid, where x_n is the output of the **conditional gradient** method,
    and where :math:`x_\star` is a minimizer of :math:`F`.
    In short, for given values of :math:`n`, :math:`L` and :math:`D`,
    :math:`\\tau(n, L, D)` is computed as the worst-case value of
    :math:`F(x_n) - F(x_\star)` when :math:`\\|x_0 - x_\star\\|^2 \\leqslant 1`.

    **Algorithm**:

        This method is presented in [1, Algorithm 1].

        .. math::
            \\begin{eqnarray}
                y_t & = & \\arg\\min_{s \\in \\mathcal{D}} \\langle s \\mid \\nabla f_1(x_t) \\rangle \\\\
                x_{t+1} & = & \\frac{t - 1}{t + 1} x_t + \\frac{2}{t + 1} y_t
            \\end{eqnarray}

    **Theoretical guarantee**:

        The **upper** guarantee obtained in [1, Theorem 1] is

        .. math:: \\tau(n, L, D) = \\frac{2LD^2}{n+2}

    References:

        `[1] Jaggi, Martin. "Revisiting Frank-Wolfe: Projection-free sparse
        convex optimization." In: Proceedings of the 30th International
        Conference on Machine Learning (ICML-13), pp. 427–435 (2013)
        <http://proceedings.mlr.press/v28/jaggi13.pdf>`_

    Args:
        L (float): the smoothness parameter.
        D (float): diameter of f2
        n (int): number of iterations.
        verbose (bool): if True, print conclusion.

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_cg_fw(L=1, D=1, n=10, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 26x26
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (0 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 2 function(s)
                 function 1 : 132 constraint(s) added
                 function 2 : 325 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.09945208318766442
        *** Example file: worst-case performance of the Conditional Gradient (Franck-Wolfe) in function value ***
            PEP-it guarantee:	 f(y_n)-f_* <= 0.0994521 ||x0 - xs||^2
            Theoretical guarantee :	 f(y_n)-f_* <= 0.166667 ||x0 - xs||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function and a convex indicator of rayon D
    func1 = problem.declare_function(function_class=SmoothConvexFunction,
                                     param={'L': L})
    func2 = problem.declare_function(function_class=ConvexIndicatorFunction,
                                     param={'D': D})
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
        lam = 2 / (i + 1)
        x = (1 - lam) * x + lam * y

    # Set the performance metric to the final distance in function values to optimum
    problem.set_performance_metric((func.value(x)) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    # when theta = 1
    theoretical_tau = 2 * L * D ** 2 / (n + 2)

    # Print conclusion if required
    if verbose:
        print('*** Example file:'
              ' worst-case performance of the Conditional Gradient (Franck-Wolfe) in function value ***')
        print('\tPEP-it guarantee:\t f(y_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee :\t f(y_n)-f_* <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))
    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_cg_fw(L=1, D=1, n=10, verbose=True)
