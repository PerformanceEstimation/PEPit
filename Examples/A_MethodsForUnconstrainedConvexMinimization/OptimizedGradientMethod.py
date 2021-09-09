from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from numpy import sqrt


def wc_ogm(L, n):
    """
    Consider the convex minimization problem
        f_* = min_x f(x),
    where f is L-smooth and convex.
    This code computes a worst-case guarantee for the optimized gradient method. That is, it computes the
    smallest possible tau(n,L) such that the guarantee
        f(x_n) - f_* <= tau(n,L) * || x_0 - x_* ||^2,
    where x_n is the output of the optimized gradient method, and where x_* is a minimizer of f.

    In short, for given values of n and L, tau(n,L) is be computed as the worst-case value of f(x_n)-f_* when
    || x_0 - x_* || == 1.

    Note that the optimized gradient method (OGM) was developed in the following two works:
    [1] Drori, Yoel, and Marc Teboulle.
     "Performance of first-order methods for smooth convex minimization: a novel approach."
     Mathematical Programming 145.1-2 (2014): 451-482.

    [2] Kim, Donghwan, and Jeffrey A. Fessler.
    "Optimized first-order methods for smooth convex minimization." Mathematical programming 159.1-2 (2016): 81-107.

    :param L: (float) the smoothness parameter.
    :param n: (int) number of iterations.
    :return:
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, {'mu': 0, 'L': L})

    # Start by defining its unique optimal point
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the GD method
    theta = [1]
    x = [x0]
    y = x0
    for i in range(n):
        x.append(y - 1 / L * func.gradient(y))
        if i < n - 1:
            theta.append( (1 + sqrt(4 * theta[i] ** 2 + 1)) / 2)
        else:
            theta.append( (1 + sqrt(8 * theta[i] ** 2 + 1)) / 2)

        y = x[i + 1] + (theta[i] - 1) / theta[i + 1] * (x[i + 1] - x[i]) + theta[i] / theta[i + 1] * (x[i + 1] - y)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func.value(y) - fs)

    # Solve the PEP
    wc = problem.solve()
    # Theoretical guarantee (for comparison)
    theory = 1/2/theta[n]**2

    print('*** Example file: worst-case performance of the optimized gradient method (OGM) in function values ***')
    print('\tPEP-it guarantee:\t f(y_n)-f_* <= ', wc)
    print('\tTheoretical guarantee:\t f(y_n)-f_* <= ', theory)
    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return wc, theory


if __name__ == "__main__":
    n = 2
    L = 1

    wc,theory = wc_ogm(L=L, n=n)

    print('{}'.format(wc))
    print('{}'.format(theory))
