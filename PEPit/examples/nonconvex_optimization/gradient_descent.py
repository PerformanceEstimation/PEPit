from PEPit.pep import PEP
from PEPit.functions.smooth_function import SmoothFunction


def wc_gradient_descent(L, gamma, n, verbose=True):
    """
    Consider the minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth.

    This code computes a worst-case guarantee for **gradient descent** with fixed step-size :math:`\\gamma`.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\gamma)` such that the guarantee

    .. math:: \\min_{t\\leqslant n} \\|\\nabla f(x_t)\\|^2 \\leqslant \\tau(n, L, \\gamma) (f(x_0) - f(x_n))

    is valid, where :math:`x_n` is the n-th iterates obtained with the gradient method with fixed step-size.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma \\nabla f(x_t),

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**:
    When :math:`\\gamma \\leqslant \\frac{1}{L}`, the **tight** theoretical guarantee can be found in [1, Theorem 1]:

    .. math:: \\min_{t\\leqslant n} \\|\\nabla f(x_t)\\|^2 \\leqslant \\frac{4}{3}\\frac{L}{n} (f(x_0) - f(x_n))

    References:

        The detailed approach is available in [1].

        [1] Polyak, Boris. "Introduction to optimization" (1987)

    Args:
        L (float): the smoothness parameter.
        gamma (float): step-size.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> L = 1
        >>> pepit_tau, theoretical_tau = wc_gradient_descent(L=L, gamma=1 / L, n=5, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 7x7
        (PEP-it) Setting up the problem: performance measure is minimum of 6 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 30 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.2666769474847614
        *** Example file: worst-case performance of gradient descent with fixed step-size ***
            PEP-it guarantee:		 min_i (f'(x_i)) ** 2 <= 0.266677 (f(x_0)-f_*)
            Theoretical guarantee:	 min_i (f'(x_i)) ** 2 <= 0.266667 (f(x_0)-f_*)

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothFunction, param={'L': L})

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value g0 and f0
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Run n steps of GD method with step-size gamma
    x = x0
    gx, fx = g0, f0

    # Set the performance metric to the minimum of the gradient norm over the iterations
    problem.set_performance_metric(gx ** 2)

    for i in range(n):
        x = x - gamma * gx
        # Set the performance metric to the minimum of the gradient norm over the iterations
        gx, fx = func.oracle(x)
        problem.set_performance_metric(gx ** 2)

    # Set the initial constraint that is the difference between fN and f0
    problem.set_initial_condition(f0 - fx <= 1)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 4 / 3 * L / n

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of gradient descent with fixed step-size ***')
        print('\tPEP-it guarantee:\t\t min_i (f\'(x_i)) ** 2 <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t min_i (f\'(x_i)) ** 2 <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    L = 1
    pepit_tau, theoretical_tau = wc_gradient_descent(L=L, gamma=1 / L, n=5, verbose=True)
