from PEPit.pep import PEP
from PEPit.Function_classes.smooth_convex_function import SmoothConvexFunction
from PEPit.Function_classes.convex_indicator import ConvexIndicatorFunction
from PEPit.Primitive_steps.linearoptimization_step import linearoptimization_step


def wc_cg_fw(L, D, n, verbose=True):
    """

    Consider the composite convex minimization problem,
        min_x { F(x) = f_1(x) + f_2(x) }
    where f_1(x) is L-smooth and convex and where f_2(x) is
    a convex indicator function of diameter at most D.

    This code computes a worst-case guarantee for the conditional Gradient method.
    That is, it computes the smallest possible tau(n,L,D) such that the guarantee
        F(x_n) - F(x_*) <= tau(n,L,D) * ||x_0 - x_*||^2
    is valid, where x_n is the output of the Conditional Gradient method, and where x_* is a minimizer of F.

    The theoretical guarantee is presented in the following reference.
    [1] Jaggi, Martin. "Revisiting Frank-Wolfe: Projection-free sparse
     convex optimization." In: Proceedings of the 30th International
     Conference on Machine Learning (ICML-13), pp. 427–435 (2013)

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param alpha: (float) parameter of the scheme.
    :param theta: (float) parameter of the scheme.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function and a convex indicator of rayon D
    func1 = problem.declare_function(SmoothConvexFunction,
                                     param={'L': L})
    func2 = problem.declare_function(ConvexIndicatorFunction,
                                     param={'D': D})
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Enforce the feasibility of x0 : there is no initial constraint on x0
    _ = func1.value(x0)
    _ = func2.value(x0)

    # Compute n steps of the Conditional Gradient / Frank-Wolfe method starting from x0
    x = x0
    for i in range(n):
        g = func1.gradient(x)
        y, _, _ = linearoptimization_step(g, func2)
        lam = 2 / (i + 1)
        x = (1 - lam) * x + lam * y

    # Set the performance metric to the final distance in function values to optimum
    problem.set_performance_metric((func.value(x)) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    # when theta = 1
    theoretical_tau = 2 * L * D ** 2 / (n + 2)

    # Print conclusion if require
    if verbose:
        print('*** Example file:'
              ' worst-case performance of the Conditional Gradient (Franck-Wolfe) in function value ***')
        print('\tPEP-it guarantee:\t f(y_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee :\t f(y_n)-f_* <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))
    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    D = 1.
    L = 1.
    n = 10

    pepit_tau, theoretical_tau = wc_cg_fw(L=L,
                                          D=D,
                                          n=n)
