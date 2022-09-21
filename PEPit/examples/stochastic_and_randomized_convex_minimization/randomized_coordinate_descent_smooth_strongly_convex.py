import numpy as np

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.point import Point


def wc_randomized_coordinate_descent_smooth_strongly_convex(L, mu, gamma, d, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex.

    This code computes a worst-case guarantee for **randomized block-coordinate descent** with step-size :math:`\\gamma`.
    That is, it computes the smallest possible :math:`\\tau(L, \\mu, \\gamma, d)` such that the guarantee

    .. math:: \\mathbb{E}_i[\\|x_{t+1}^{(i)} - x_\star \\|^2] \\leqslant \\tau(L, \\mu, \\gamma, d) \\|x_{t} - x_\\star\\|^2
    
    where :math:`x_{t+1}^{(i)}` denotes the value of the iterate :math:`x_{t+1}` in the scenario
    where the :math:`i` th block of coordinates is selected for the update  with fixed step-size
    :math:`\\gamma`, :math:`d` is the number of blocks of coordinates and where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`\\mu`, :math:`L`, :math:`d`, and :math:`\\gamma`, :math:`\\tau(L, \\mu, \\gamma, d)` is
    computed as the worst-case value of :math:`\\mathbb{E}_i[\\|x_{t+1}^{(i)} - x_\star \\|^2]` when
    :math:`\\|x_t - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    Randomized block-coordinate descent is described by

    .. math::
        \\begin{eqnarray}
            \\text{Pick random }i & \\sim & \\mathcal{U}\\left([|1, d|]\\right), \\\\
            x_{t+1}^{(i)} & = & x_t - \\gamma \\nabla_i f(x_t),
        \\end{eqnarray}

    where :math:`\\gamma` is a step-size and :math:`\\nabla_i f(x_t)` is the partial derivative corresponding to the block :math:`i`.

    **Theoretical guarantee**:
    When :math:`\\gamma \\leqslant \\frac{1}{L}`, the **tight** theoretical guarantee can be found in [1, Appendix I, Theorem 17]:

    .. math:: \\mathbb{E}_i[\\|x_{t+1}^{(i)} - x_\star \\|^2] \\leqslant \\rho^2 \\|x_t-x_\\star\\|^2,

    where :math:`\\rho^2 = \\max \\left( \\frac{(\\gamma\\mu - 1)^2 + d - 1}{d},\\frac{(\\gamma L - 1)^2 + d - 1}{d} \\right)`.

    **References**:

    `[1] A. Taylor, F. Bach (2021). Stochastic first-order methods: non-asymptotic and computer-aided
    analyses via potential functions. In Conference on Learning Theory (COLT).
    <https://arxiv.org/pdf/1902.00947.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong-convexity parameter.
        gamma (float): the step-size.
        d (int): the dimension.
        verbose (int): Level of information details to print.

                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> L = 1
        >>> mu = 0.1
        >>> gamma = 2 / (mu + L)
        >>> pepit_tau, theoretical_tau = wc_randomized_coordinate_descent_smooth_strongly_convex(L=L, mu=mu, gamma=gamma, d=2, verbose=1)
        (PEPit) Setting up the problem: size of the main PSD matrix: 4x4
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (3 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                         function 1 : Adding 2 scalar constraint(s) ...
                         function 1 : 2 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.8347107377149059
        *** Example file: worst-case performance of randomized coordinate gradient descent ***
                PEPit guarantee:         E||x_(n+1) - x_*||^2 <= 0.834711 ||x_n - x_*||^2
                Theoretical guarantee:   E||x_(n+1) - x_*||^2 <= 0.834711 ||x_n - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Define an orthogonal decomposition of the gradient (into a partition of the space)
    g0 = func.gradient(x0)
    gradients = []
    for i in range(d - 1):
        gradients.append(Point())
    gd = g0 - np.sum(gradients)  # Define the last point as a function of the whole gradient and past iterates
    gradients.append(gd)
    # Add orthogonality constraints
    for i in range(d):
        for j in range(d):
            if i != j:
                problem.add_constraint(gradients[i] * gradients[j] == 0)

    # Compute the expectation of randomized coordinate descent step
    x = x0
    var = np.mean([(x - gamma * grad - xs) ** 2 for grad in gradients])

    # Set the performance metric to the variance
    problem.set_performance_metric(var)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = max(((mu * gamma - 1) ** 2 + d - 1) / d, ((L * gamma - 1) ** 2 + d - 1) / d)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of randomized coordinate gradient descent ***')
        print('\tPEPit guarantee:\t E||x_(n+1) - x_*||^2 <= {:.6} ||x_n - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t E||x_(n+1) - x_*||^2 <= {:.6} ||x_n - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    mu = 0.1
    gamma = 2 / (mu + L)
    pepit_tau, theoretical_tau = wc_randomized_coordinate_descent_smooth_strongly_convex(L=L, mu=mu, gamma=gamma, d=2, verbose=1)
