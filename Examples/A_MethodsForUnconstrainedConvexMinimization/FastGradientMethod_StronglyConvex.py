from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from numpy import sqrt


def wc_fgm(mu, L, n):
    """
    Consider the convex minimization problem
        f_* = min_x f(x),
    where f is L-smooth and mu-strongly-convex.

    This code computes a worst-case guarantee for the fast gradient method, a.k.a. accelerated gradient method. That is, it computes the
    smallest possible tau(n, mu, L) such that the guarantee
        f(x_n) - f_* <= tau(n, mu, L) (f(x_0) -  f(x_*) +  mu/2*|| x_0 - x_* ||**2),
    is valid, where x_n is the output of the optimized gradient method, and where x_* is a minimizer of f.

    In short, for given values of n and L, tau(n,L) is be computed as the worst-case value of f(x_n)-f_* when
    f(x_0) -  f(x_*) +  mu/2 * || x_0 - x_* ||**2 == 1.

    Theoretical rates can be found in the following paper
    For an Upper bound (not tight):
    [1] Acceleration Methods, Monograph, Alexandre d’Aspremont, Damien Scieur, Adrien Taylor, https://arxiv.org/pdf/2101.09545.pdf

    :param mu: (float) the strong-convexity parameter.
    :param L: (float) the smoothness parameter.
    :param n: (int) number of iterations.
    :return:
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, {'mu': mu, 'L': L})

    # Start by defining its unique optimal point
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition(func.value(x0)- fs + + mu/2 * (x0 - xs) ** 2 <= 1)

    # Run the GD method
    kappa = mu/L
    x_new = x0
    y = x0
    for i in range(n):
        x_old = x_new
        x_new = y - 1/L * func.gradient(y)
        y = x_new + (1-sqrt(kappa))/(1+sqrt(kappa))* (x_new-x_old)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func.value(x_new) - fs)

    # Solve the PEP
    wc = problem.solve()
    # Theoretical guarantee (for comparison)
    if mu>0:
        theory = (1-sqrt(kappa))**n # see [1], Corollary 4.15,
    else:
        theory = 0
        print("Momentum is here designed for strongly convex functions")

    print('*** Example file: worst-case performance of the Fast Gradient Method (FGM) in function values (initial '
          'condition: f(x_0) -  f(x_*) +  mu/2 * || x_0 - x_* ||**2 <= 1)***')
    print('\tPEP-it guarantee:\t f(y_n)-f_* <= ', wc)
    print('\tTheoretical guarantee:\t f(y_n)-f_* <= ', theory)
    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return wc, theory


if __name__ == "__main__":
    n = 1
    L = 1
    mu = 0.1

    wc,theory = wc_fgm(mu=mu, L=L, n=n)

    print('{}'.format(wc))
    print('{}'.format(theory))
