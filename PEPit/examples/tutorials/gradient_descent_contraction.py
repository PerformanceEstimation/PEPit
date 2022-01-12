from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def wc_gradient_descent_contraction(L, mu, gamma, n, verbose=True):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for **gradient descent** with fixed step-size :math:`\\gamma`.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu, \\gamma)` such that the guarantee

    .. math:: \\| x_n - y_n \\|^2 \\leqslant \\tau(n, L, \\mu, \\gamma) \\| x_0 - y_0 \\|^2

    is valid, where :math:`x_n` and :math:`y_n` are the outputs of
    the gradient descent method with fixed step-size :math:`\\gamma`,
    starting respectively from :math:`x_0` and :math:`y_0`.

    In short, for given values of :math:`n`, :math:`L`, :math:`\\mu` and :math:`\\gamma`,
    :math:`\\tau(n, L, \\mu \\gamma)` is computed as the worst-case value of :math:`\\| x_n - y_n \\|^2`
    when :math:`\\| x_0 - y_0 \\|^2 \\leqslant 1`.

    **Algorithm**:
    For :math:`t\\in\\{0,1,\\ldots,n-1\\}`, gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma \\nabla f(x_t),

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**:
    The **tight** theoretical guarantee is

    .. math:: \\| x_n - y_n \\|^2 \\leqslant  \\max\\{(1-L\\gamma)^2,(1-\\mu \\gamma)^2\\}^n\\| x_0 - y_0 \\|^2,
    
    which is tight on simple quadratic functions.

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong-convexity parameter.
        gamma (float): step-size.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> L = 1
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_contraction(L=L, mu=0.1, gamma=1 / L, n=1, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 4x4
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 2 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: MOSEK); optimal value: 0.8100000029204689
        *** Example file: worst-case performance of gradient descent with fixed step-sizes in contraction ***
            PEP-it guarantee:		 ||x_n - y_n||^2 <= 0.81 ||x_0 - y_0||^2
            Theoretical guarantee:	 ||x_n - y_n||^2 <= 0.81 ||x_0 - y_0||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})

    # Then define the starting points x0 and y0 of the algorithm
    x_0 = problem.set_initial_point()
    y_0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and y0
    problem.set_initial_condition((x_0 - y_0) ** 2 <= 1)

    # Run n steps of the GD method
    x = x_0
    y = y_0
    for _ in range(n):
        x = x - gamma * func.gradient(x)
        y = y - gamma * func.gradient(y)

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric((x - y) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = max((1 - gamma * L) ** 2, (1 - gamma * mu) ** 2) ** n

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of gradient descent with fixed step-sizes in contraction ***')
        print('\tPEP-it guarantee:\t ||x_n - y_n||^2 <= {:.6} ||x_0 - y_0||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||x_n - y_n||^2 <= {:.6} ||x_0 - y_0||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    L = 1
    pepit_tau, theoretical_tau = wc_gradient_descent_contraction(L=L, mu=0.1, gamma=1 / L, n=1, verbose=True)
