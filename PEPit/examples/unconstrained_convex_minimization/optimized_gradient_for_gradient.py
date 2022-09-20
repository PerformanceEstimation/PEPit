from math import sqrt

from PEPit import PEP
from PEPit.functions import SmoothConvexFunction


def wc_optimized_gradient_for_gradient(L, n, verbose=1):
    """
    Consider the minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and convex.

    This code computes a worst-case guarantee for **optimized gradient method for gradient** (OGM-G).
    That is, it computes the smallest possible :math:`\\tau(n, L)` such that the guarantee

    .. math:: \\|\\nabla f(x_n)\\|^2 \\leqslant \\tau(n, L) (f(x_0) - f_\\star)

    is valid, where :math:`x_n` is the output of OGM-G and where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n` and :math:`L`, :math:`\\tau(n, L)` is computed as the worst-case value
    of :math:`\\|\\nabla f(x_n)\\|^2` when :math:`f(x_0)-f_\\star \\leqslant 1`.

    **Algorithm**:
    For :math:`t\\in\\{0,1,\\ldots,n-1\\}`, the optimized gradient method for gradient [1, Section 6.3] is described by

        .. math::
            :nowrap:

            \\begin{eqnarray}
                y_{t+1} & = & x_t - \\frac{1}{L} \\nabla f(x_t),\\\\
                x_{t+1} & = & y_{t+1} + \\frac{(\\tilde{\\theta}_t-1)(2\\tilde{\\theta}_{t+1}-1)}{\\tilde{\\theta}_t(2\\tilde{\\theta}_t-1)}(y_{t+1}-y_t)+\\frac{2\\tilde{\\theta}_{t+1}-1}{2\\tilde{\\theta}_t-1}(y_{t+1}-x_t),
            \\end{eqnarray}

    with

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\tilde{\\theta}_n & = & 1 \\\\
                \\tilde{\\theta}_t & = & \\frac{1 + \\sqrt{4 \\tilde{\\theta}_{t+1}^2 + 1}}{2}, \\forall t \\in [|1, n-1|] \\\\
                \\tilde{\\theta}_0 & = & \\frac{1 + \\sqrt{8 \\tilde{\\theta}_{1}^2 + 1}}{2}.
            \\end{eqnarray}

    **Theoretical guarantee**:
    The **tight** worst-case guarantee can be found in [1, Theorem 6.1]:

    .. math:: \\|\\nabla f(x_n)\\|^2 \\leqslant \\frac{2L(f(x_0)-f_\\star)}{\\tilde{\\theta}_0^2},

    where tightness is achieved on Huber losses, see [1, Section 6.4].

    **References**:
    The optimized gradient method for gradient was developed in [1].

    `[1] D. Kim, J. Fessler (2021).
    Optimizing the efficiency of first-order methods for decreasing the gradient of smooth convex functions.
    Journal of optimization theory and applications, 188(1), 192-219.
    <https://arxiv.org/pdf/1803.06600.pdf>`_

    Args:
        L (float): the smoothness parameter.
        n (int): number of iterations.
        verbose (int): Level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_optimized_gradient_for_gradient(L=3, n=4, verbose=1)
        (PEPit) Setting up the problem: size of the main PSD matrix: 7x7
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                         function 1 : Adding 30 scalar constraint(s) ...
                         function 1 : 30 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.30700758289614183
        *** Example file: worst-case performance of optimized gradient method for gradient ***
                PEP-it guarantee:        ||f'(x_n)||^2 <= 0.307008 (f(x_0) - f_*)
                Theoretical guarantee:   ||f'(x_n)||^2 <= 0.307007 (f(x_0) - f_*)

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define x0 the starting point of the algorithm and its function value f(x_0)
    x0 = problem.set_initial_point()
    f0 = func(x0)

    # Set the initial constraint that is f(x_0) - f(x_*)
    problem.set_initial_condition(f0 - fs <= 1)

    # Compute scalar sequence of \tilde{theta}_t
    theta_tilde = [1]  # compute \tilde{theta}_{t} from \tilde{theta}_{t+1} (sequence in reverse order)
    for i in range(n):
        if i < n - 1:
            theta_tilde.append((1 + sqrt(4 * theta_tilde[i] ** 2 + 1)) / 2)
        else:
            theta_tilde.append((1 + sqrt(8 * theta_tilde[i] ** 2 + 1)) / 2)
    theta_tilde.reverse()

    # Run n steps of the optimized gradient method for gradient (OGM-G) method
    x = x0
    y_new = x0

    for i in range(n):
        y_old = y_new
        y_new = x - 1 / L * func.gradient(x)
        x = y_new + (theta_tilde[i] - 1) * (2 * theta_tilde[i + 1] - 1) / theta_tilde[i] / (2 * theta_tilde[i] - 1) \
            * (y_new - y_old) + (2 * theta_tilde[i + 1] - 1) / (2 * theta_tilde[i] - 1) * (y_new - x)

    # Set the performance metric to the gradient norm
    problem.set_performance_metric(func.gradient(x) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 2 * L / (theta_tilde[0] ** 2)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of optimized gradient method for gradient ***')
        print('\tPEP-it guarantee:\t ||f\'(x_n)||^2 <= {:.6} (f(x_0) - f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||f\'(x_n)||^2 <= {:.6} (f(x_0) - f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_optimized_gradient_for_gradient(L=3, n=4, verbose=1)
