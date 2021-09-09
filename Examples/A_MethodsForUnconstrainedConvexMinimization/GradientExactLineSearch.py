from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Primitive_steps.exactlinesearch_step import exactlinesearch_step

def wc_ELS(L, mu, n):
    """
    Consider the convex minimization problem
        f_* = min_x f(x),
    where f is L-smooth and mu-strongly convex.
    This code computes a worst-case guarantee for the gradient method with exact linesearch (ELS). That is, it computes
    the smallest possible tau(n,L,mu) such that the guarantee
        f(x_n) - f_* <= tau(n,L,mu) * ( f(x_0) - f_* )
    is valid, where x_n is the output of the gradient descent with exact linesearch, and where x_* is the minimizer of f.

    In short, for given values of n and L, tau(n,L) is be computed as the worst-case value of f(x_n)-f_* when
    f(x_0) - f_* == 1.

    The detailed approach (based on convex relaxations) is available in
    [1] De Klerk, Etienne, François Glineur, and Adrien B. Taylor.
    "On the worst-case complexity of the gradient method with exact line search for smooth strongly convex functions."
    Optimization Letters (2017).

    The tight guarantee obtained in [1] is tau(n,L,mu) = ((L-mu)/(L+mu))**(2*n).

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
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
    g0,f0 = func.oracle(x0)

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition( f0 - fs <= 1)

    # Run the GD method with ELS
    x = x0
    gx = g0
    for i in range(n):
        x,gx,fx = exactlinesearch_step(x,func,[gx])

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func.value(x) - fs)

    # Solve the PEP
    wc = problem.solve()
    # Theoretical guarantee (for comparison)
    theory = ((L-mu)/(L+mu))**(2*n)

    print('*** Example file: worst-case performance of gradient descent with exact linesearch (ELS) ***')
    print('\tPEP-it guarantee:\t f(y_n)-f_* <= ', wc)
    print('\tTheoretical guarantee:\t f(y_n)-f_* <= ', theory)
    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return wc, theory


if __name__ == "__main__":
    n = 2
    L = 1
    mu= .1

    wc,theory = wc_ELS(L=L, mu=mu, n=n)

    print('{}'.format(wc))
    print('{}'.format(theory))
