from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_pgd(L, mu, gamma, n, verbose=True):
    """
    Consider the convex minimization problem
        f_* = min_x f1(x) + f2(x),
    where f is L-smooth and mu-strongly convex, and where f2 is a closed convex and proper.

    This code computes a worst-case guarantee for the proximal gradient method.
    That is, the code computes the smallest possible tau(n,L,mu) such that the guarantee
        f(x_n) - f_* <= tau(n,L,mu) * (f(x_0) - f_*),
    is valid, where x_n is the output of the proximal gradient, and where x_* is a minimizer of f.

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity paramter
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function and a closed convex proper function
    f1 = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})
    f2 = problem.declare_function(ConvexFunction, param={})
    func = f1 + f2

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the proximal gradient method starting from x0
    x = x0
    for _ in range(n):
        y = x - gamma * f1.gradient(x)
        x, _, _ = proximal_step(y, f2, gamma)

    # Set the performance metric to the distance between x and xs
    problem.set_performance_metric((x - xs) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = max((1 - mu*gamma)**2, (1 - L*gamma)**2)**n

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Proximal Gradient Method in function values***')
        print('\tPEP-it guarantee:\t f(y_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee :\t f(y_n)-f_* <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
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
