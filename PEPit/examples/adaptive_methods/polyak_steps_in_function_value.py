from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def wc_polyak_steps_in_function_value(L, mu, gamma, verbose=True):
    """
    Consider the minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex, and :math:`x_\\star=\\arg\\min_x f(x)`.

    This code computes a worst-case guarantee for a variant of a **gradient method** relying on **Polyak step-sizes**.
    That is, it computes the smallest possible :math:`\\tau(L, \\mu, \\gamma)` such that the guarantee

    .. math:: f(x_{t+1}) - f_\\star \\leqslant \\tau(L, \\mu, \\gamma) (f(x_t) - f_\\star)

    is valid, where :math:`x_t` is the output of the gradient method with PS and :math:`\\gamma` is the effective value
    of the step-size of the gradient method.

    In short, for given values of :math:`L`, :math:`\\mu`, and :math:`\\gamma`, :math:`\\tau(L, \\mu, \\gamma)` is computed as the worst-case
    value of :math:`f(x_{t+1})-f_\\star` when :math:`f(x_t)-f_\\star \\leqslant 1`.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma \\nabla f(x_t),

    where :math:`\\gamma` is a step-size. The Polyak step-size rule under consideration here corresponds to choosing
    of :math:`\\gamma` satisfying:

    .. math:: \\|\\nabla f(x_t)\\|^2 = 2  L (2 - \\gamma) (f(x_t) - f_\\star).

    **Theoretical guarantee**:
    The gradient method with the variant of Polyak step-sizes under consideration enjoys the
    **tight** theoretical guarantee [1, Proposition 2]:

    .. math:: f(x_{t+1})-f_\\star \\leqslant  \\tau(L,\\mu,\\gamma) (f(x_{t})-f_\\star),

    where :math:`\\gamma` is the effective step-size used at iteration :math:`t` and

    .. math::
            :nowrap:

            \\begin{eqnarray}
                \\tau(L,\\mu,\\gamma) & = & \\left\\{\\begin{array}{ll} (\\gamma L - 1)  (L \\gamma  (3 - \\gamma (L + \\mu)) - 1)  & \\text{if } \\gamma\in[\\tfrac{1}{L},\\tfrac{2L-\mu}{L^2}],\\\\
                0 & \\text{otherwise.} \\end{array}\\right.
            \\end{eqnarray}

    **References**:

    `[1] M. Barré, A. Taylor, A. d’Aspremont (2020). Complexity guarantees for Polyak steps with momentum.
    In Conference on Learning Theory (COLT).
    <https://arxiv.org/pdf/2002.00915.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        gamma (float): the step-size.
        verbose (bool): if True, print conclusion

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> L = 1
        >>> mu = 0.1
        >>> gamma = 2 / (L + mu)
        >>> pepit_tau, theoretical_tau = wc_polyak_steps_in_function_value(L=L, mu=mu, gamma=gamma, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 4x4
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (2 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 6 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.6694215432773613
        *** Example file: worst-case performance of Polyak steps ***
            PEPit guarantee:		 f(x_1) - f_* <= 0.669422 (f(x_0) - f_*)
            Theoretical guarantee:	 f(x_1) - f_* <= 0.669421 (f(x_0) - f_*)

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, param={'L': L, 'mu': mu})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value gn and fn
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Set the initial condition to the distance betwenn x0 and xs
    problem.set_initial_condition(f0 - fs <= 1)

    # Run the Polayk steps at iteration 1
    x1 = x0 - gamma * g0
    g1, f1 = func.oracle(x1)

    # Set the initial condition to the Polyak step-size
    problem.set_initial_condition(g0 ** 2 == 2 * L * (2 - gamma) * (f0 - fs))

    # Set the performance metric to the distance in function values between x_1 and x_* = xs
    problem.set_performance_metric(f1 - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    if gamma >= 1/L and gamma <= (2*L-mu)/L**2:
        theoretical_tau = (gamma * L - 1) * (L * gamma * (3 - gamma * (L + mu)) - 1)
    else:
        theoretical_tau = 0.

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of Polyak steps ***')
        print('\tPEPit guarantee:\t f(x_1) - f_* <= {:.6} (f(x_0) - f_*) '.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_1) - f_* <= {:.6} (f(x_0) - f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    L = 1
    mu = 0.1
    gamma = 2 / (L + mu)
    pepit_tau, theoretical_tau = wc_polyak_steps_in_function_value(L=L, mu=mu, gamma=gamma, verbose=True)
