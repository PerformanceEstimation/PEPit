from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_pgd(L, mu, gamma, n, verbose=True):
    """
    Consider the convex minimization problem
        f_* = min_x f1(x) + f2(x),
    where f2 is L-smooth and mu-strongly convex, and where f2 is a closed convex and proper.
    Instead of declaring f1 and f2, let us declare:
        - F1 mu/3-strongly-convex and L/3 smooth,
        - F2 mu/2-strongly-convex and L/2 smooth,
        - F3 and F4 two closed proper convex functions.

    This code computes a worst-case guarantee for the proximal gradient method, for :
        f = f1 + f2,
    with f1 = 5*F3 + 2*F4 and f2 = (3*F1 + 2*F2)/2.

    That is, the code computes the smallest possible tau(n,L,mu) such that the guarantee
       ||x_n - x_*||^2 <= tau(n,L,mu) * ||x_0 - x_*||^2,
    is valid, where x_n is the output of the proximal gradient, and where x_* is a minimizer of f.

    The worst-case bound obtained in this example should match both the theoretical upper bound, and the
    PEPit bound obtained for the Proximal Gradient when considering directly f1 L-smoth and mu-stronlgy convex,
    and f2 closed proper and convex.

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param gamma: (float) the step size.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    F1 = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu / 3, 'L': L / 3})
    F2 = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu / 2, 'L': L / 2})
    f3 = problem.declare_function(ConvexFunction, param={})
    f4 = problem.declare_function(ConvexFunction, param={})
    f2 = 5 * f3 + 2 * f4
    f1 = (3 * F1 + 2 * F2) / 2
    func = f1 + f2
    # Start by defining its unique optimal point
    xs = func.optimal_point()

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
