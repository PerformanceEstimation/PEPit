from PEPit import PEP
from PEPit.functions import ConvexFunction
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.primitive_steps import proximal_step


def wc_proximal_gradient_complexified2(L, mu, gamma, n, verbose=1):
    """
    See description in `PEPit/examples/unconstrained_convex_minimization/proximal_point.py`.
    This example is for testing purposes; the worst-case result is supposed to be the same as that of the other routine,
    but the parameterization is different (convex function to be minimized is explicitly formed as a sum of four convex
    functions). That is, the minimization problem is the composite convex minimization problem

    .. math:: f_\\star = \\min_x \\{f(x) = f_1(x) + f_2(x)\\},

    where :math:`f_1` is :math:`L`-smooth and :math:`\\mu`-strongly convex,
    and where :math:`f_2` is closed convex and proper.
    We further set :math:`f_1 = \\frac{3 F_1 + 2 F_2}{2}` and :math:`f_2 = 5 F_3 + 2 F_4` where

    .. math::
        \\begin{eqnarray}
            F_1 & \\text{ is } & \\frac{\\mu}{3}\\text{-strongly convex and } \\frac{L}{3}\\text{-smooth} \\\\
            F_2 & \\text{ is } & \\frac{\\mu}{2}\\text{-strongly convex and } \\frac{L}{2}\\text{-smooth} \\\\
            F_3 & \\text{ is } & \\text{closed proper and convex.} \\\\
            F_4 & \\text{ is } & \\text{closed proper and convex.}
        \\end{eqnarray}

    Args
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        gamma (float): the step size.
        n (int): number of iterations.
        verbose (int): Level of information details to print.
                       -1: No verbose at all.
                       0: This example's output.
                       1: This example's output + PEPit information.
                       2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_proximal_gradient_complexified(L=1, mu=.1, gamma=1, n=2, verbose=1)
        (PEPit) Setting up the problem: size of the main PSD matrix: 13x13
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 4 function(s)
                         function 1 : Adding 6 scalar constraint(s) ...
                         function 1 : 6 scalar constraint(s) added
                         function 2 : Adding 6 scalar constraint(s) ...
                         function 2 : 6 scalar constraint(s) added
                         function 3 : Adding 6 scalar constraint(s) ...
                         function 3 : 6 scalar constraint(s) added
                         function 4 : Adding 6 scalar constraint(s) ...
                         function 4 : 6 scalar constraint(s) added
        (PEPit) Setting up the problem: 1 partition(s) added
                         partition 1 with 6 blocks: Adding 0 scalar constraint(s)...
                         partition 1 with 6 blocks: 0 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.6561016829295551
        *** Example file: worst-case performance of gradient descent ***
                PEPit guarantee:         ||x_n-x_*||^2 <= 0.656102 ||x_0-x_*||^2
                Theoretical guarantee:   ||x_n-x_*||^2 <= 0.6561 ||x_0-x_*||^2

    """

    # Instantiate PEP
    problem = PEP()
    partition = problem.declare_block_partition(d=6)

    # Declare strongly convex smooth functions
    smooth_strongly_convex_1 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu / 3, L=L / 3)
    smooth_strongly_convex_2 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu / 2, L=L / 2)

    # Declare convex smooth functions
    convex_1 = problem.declare_function(ConvexFunction)
    convex_2 = problem.declare_function(ConvexFunction)

    f1 = (3 * smooth_strongly_convex_1 + 2 * smooth_strongly_convex_2) / 2
    f2 = 5 * convex_1 + 2 * convex_2
    func = f1 + f2

    # Start by defining its unique optimal point
    xs = func.stationary_point()
    _ = partition.get_block(xs,1) #useless partition

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()
    _ = partition.get_block(x0,1) #useless partition

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the GD method
    x = x0
    for _ in range(n):
        y = x - gamma * f1.gradient(x)
        x, _, _ = proximal_step(y, f2, gamma)

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric((x - xs) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = max((1 - gamma * mu) ** 2, (1 - gamma * L) ** 2) ** n

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent ***')
        print('\tPEPit guarantee:\t ||x_n-x_*||^2 <= {:.6} ||x_0-x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||x_n-x_*||^2 <= {:.6} ||x_0-x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_proximal_gradient_complexified(L=1, mu=.1, gamma=1, n=2, verbose=1)
