from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.functions.convex_function import ConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_fgm(mu, L, n, verbose=True):
    """
    Consider the convex minimization problem
        F_\star = min_x { F(x) = f(x) + h(x) },
    where f is L-smooth and convex, and where h is closed convex and proper.
    We further consider the case where the gradient of f can be evaluated, and where the proximal operator
    of h can be evaluated as well.

    This code computes a worst-case guarantee for the fast proximal gradient method (a.k.a. accelerated gradient).
    That is, the code computes the smallest possible tau(mu,L,n) such that the guarantee
        f(x_n) - f_\star <= tau(mu,L,n) * || x_0 - x_\star ||^2,
    is valid, where x_n is the output of the fast proximal gradient, and where x_\star is a minimizer of f.

    In short, for given values of n and L, tau(n,L) is be computed as the worst-case value of f(x_n)-f_\star when
    || x_0 - x_\star || == 1.

    Theoretical rates can be found in the following paper
    For an Upper bound (not tight)
    [1] A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems∗
    Amir Beck† and Marc Teboulle‡

    For an exact bound (convex):
    [2] Exact Worst-case Performance of First-order Methods for Composite Convex Optimization
    Adrien B. Taylor, Julien M. Hendrickx, François Glineur

    :param L: (float) the smoothness parameter.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function and a convex function
    f = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})
    h = problem.declare_function(ConvexFunction, param={})
    F = f + h

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = F.stationary_point()
    Fs = F.value(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute n steps of the Fast Proximal Gradient method starting from x0
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
        print('*** Example file: worst-case performance of the Fast Proximal Gradient Method in function values***')
        print('\tPEP-it guarantee:\t f(y_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee :\t f(y_n)-f_* <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 1
    mu = 0
    L = 1

    pepit_tau, theoretical_tau = wc_fgm(mu=mu,
                                        L=L,
                                        n=n)
