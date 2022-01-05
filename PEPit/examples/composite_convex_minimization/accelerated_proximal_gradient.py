from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.functions import ConvexFunction
from PEPit.primitive_steps import proximal_step


def wc_accelerated_proximal_gradient(mu, L, n, verbose=True):
    """
    Consider the composite convex minimization problem

    .. math:: F_\\star \\triangleq \\min_x \\{F(x) \equiv f(x) + h(x)\\},

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex,
    and where :math:`h` is closed convex and proper.

    This code computes a worst-case guarantee for the **accelerated proximal gradient** method,
    also known as **fast proximal gradient (FPGM)** method.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu)` such that the guarantee

    .. math :: F(x_n) - F(x_\\star) \\leqslant \\tau(n, L, \\mu) \\|x_0 - x_\\star\\|^2,

    is valid, where :math:`x_n` is the output of the **accelerated proximal gradient** method,
    and where :math:`x_\\star` is a minimizer of :math:`F`.

    In short, for given values of :math:`n`, :math:`L` and :math:`\\mu`,
    :math:`\\tau(n, L, \\mu)` is computed as the worst-case value of
    :math:`F(x_n) - F(x_\\star)` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**: Accelerated proximal gradient is described as follows, for :math:`t \in \\{ 0, \\dots, n-1\\}`,

    .. math::
        :nowrap:

        \\begin{eqnarray}
            x_{t+1} & = & \\arg\\min_x \\left\\{h(x)+\\frac{L}{2}\|x-\\left(y_{t} - \\frac{1}{L} \\nabla f(y_t)\\right)\\|^2 \\right\\}, \\\\
            y_{t+1} & = & x_{t+1} + \\frac{i}{i+3} (x_{t+1} - x_{t}),
        \\end{eqnarray}

    where :math:`y_{0} = x_0`.

    **Theoretical guarantee**: A **tight** (empirical) worst-case guarantee for FPGM is obtained in [1, method FPGM1 in Sec. 4.2.1, Table 1 in sec 4.2.2], for :math:`\\mu=0`:

    .. math:: F(x_n) - F_\\star \\leqslant \\frac{2 L}{n^2+5n+2} \\|x_0 - x_\\star\\|^2,

    which is attained on simple one-dimensional constrained linear optimization problems.

    **References**:

    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017). Exact worst-case performance of first-order methods for composite
    convex optimization. SIAM Journal on Optimization, 27(3):1283–1313.
    <https://arxiv.org/pdf/1512.07516.pdf>`_


    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (float): theoretical value.

    Example:
        >>> pepit_tau, theoretical_tau = wc_accelerated_proximal_gradient(L=1, mu=0, n=4, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 6x6
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
                 function 1 : 6 constraint(s) added
                 function 2 : 2 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        *** Example file: worst-case performance of the Fast Proximal Gradient Method in function values***
            PEPit guarantee:       f(x_n)-f_* <= 0.0526302 ||x0 - xs||^2
            Theoretical guarantee:  f(x_n)-f_* <= 0.0526316 ||x0 - xs||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function and a convex function
    f = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})
    h = problem.declare_function(ConvexFunction, param={})
    F = f + h

    # Start by defining its unique optimal point xs = x_* and its function value Fs = F(x_*)
    xs = F.stationary_point()
    Fs = F.value(xs)

    # Then define the starting point x0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute n steps of the accelerated proximal gradient method starting from x0
    x_new = x0
    y = x0
    for i in range(n):
        x_old = x_new
        x_new, _, hx_new = proximal_step(y - 1 / L * f.gradient(y), h, 1 / L)
        y = x_new + i / (i + 3) * (x_new - x_old)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric((f.value(x_new) + hx_new) - Fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    if mu == 0:
        theoretical_tau = 2 * L / (n ** 2 + 5 * n + 2)  # tight, see [2], Table 1 (column 1, line 1)
    else:
        theoretical_tau = 2 * L / (n ** 2 + 5 * n + 2)  # not tight (bound for smooth convex functions)
        print('Warning: momentum is tuned for non-strongly convex functions.')

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Accelerated Proximal Gradient Method in function values***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee :\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_accelerated_proximal_gradient(L=1, mu=0, n=4, verbose=True)
