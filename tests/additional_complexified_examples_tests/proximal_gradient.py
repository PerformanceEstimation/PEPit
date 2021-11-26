from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.functions.convex_function import ConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_pgd(L, mu, gamma, n, verbose=True):
    """
    See description in Examples/a_methods_for_unconstrained_convex_minimization/proximal_point_method.py.
    This example is for testing purposes; the worst-case result is supposed to be the same as that of the other routine,
    but the parameterization is different (convex function to be minimized is explicitly formed as a sum of four convex
    functions). That is, the minimization problem is the composite convex minimization problem

    .. math:: f_\star = \\min_x \\{f(x) = f_1(x) + f_2(x)\\},

    where :math:`f_1` is :math:`L`-smooth and :math:`\\mu`-strongly convex, and where :math:`f_2` is closed convex and
    proper. We further let :math:`f_1=(3 F_1+2F_2)/2` and :math:`f_2=5 F_2+2F_4`
        - F1 mu/3-strongly convex and L/3 smooth,
        - F2 mu/2-strongly convex and L/2 smooth,
        - F3 and F4 two closed proper convex functions.

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param gamma: (float) the step size.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare strongly convex smooth functions
    smooth_strongly_convex_1 = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu / 3, 'L': L / 3})
    smooth_strongly_convex_2 = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu / 2, 'L': L / 2})

    # Declare convex smooth functions
    smooth_convex_1 = problem.declare_function(ConvexFunction, param={})
    smooth_convex_2 = problem.declare_function(ConvexFunction, param={})

    f1 = (3 * smooth_strongly_convex_1 + 2 * smooth_strongly_convex_2) / 2
    f2 = 5 * smooth_convex_1 + 2 * smooth_convex_2
    func = f1 + f2

    # Start by defining its unique optimal point
    xs = func.stationary_point()

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

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
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = max((1 - gamma * mu) ** 2, (1 - gamma * L) ** 2) ** n

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of gradient descent ***')
        print('\tPEP-it guarantee:\t\t ||x_n-x_*||^2 <= {:.6} ||x_0-x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||x_n-x_*||^2 <= {:.6} ||x_0-x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 2
    L = 1
    mu = .1
    gamma = 1

    pepit_tau, theoretical_tau = wc_pgd(L=L,
                                        mu=mu,
                                        gamma=gamma,
                                        n=n)
