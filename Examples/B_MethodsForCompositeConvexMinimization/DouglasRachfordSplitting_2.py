from PEPit.pep import PEP
from PEPit.Function_classes.smooth_convex_function import SmoothConvexFunction
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_drs_2(L, alpha, theta, n, verbose=True):
    """
    Consider the composite convex minimization problem,
        min_x { F(x) = f_1(x) + f_2(x) }
    where f_1(x) is L-smooth and mu-strongly convex, and f_2 is convex,
    closed and proper. Both proximal operators are assumed to be available.

    This code computes a worst-case guarantee for the Douglas Rachford Splitting (DRS) method,
    where our notations for the DRS algorithm are as follows:
        x_k     = prox_{\alpha f2}(w_k)
        y_k     = prox_{\alpha f1}(2*x_k-w_k)
        w_{k+1} = w_k +\theta (y_k - x_k)

    That is, it computes the smallest possible tau(n,L) such that the guarantee
        F(y_n) - F(x_*) <= tau(n,L) * ||x_0 - x_*||^2.
    is valid, where it is known that xk and yk converge to xs, but not wk, and hence
    we require the initial condition on x0 (arbitrary choice; partially justified by
    the fact we choose f2 to be the smooth function). Note that yN is feasible as it
    has a finite value for f1 (output of the proximal operator on f1) and as f2 is smooth.

    :param L: (float) the smoothness parameter.
    :param alpha: (float) parameter of the scheme.
    :param theta: (float) parameter of the scheme.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex and a smooth strongly convex function.
    func1 = problem.declare_function(ConvexFunction,
                                     param={})
    func2 = problem.declare_function(SmoothConvexFunction,
                                     param={'L': L})
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()
    _ = func.value(x0)

    # Compute n steps of the Douglas Rachford Splitting starting from x0
    x = [x0 for _ in range(n)]
    w = [x0 for _ in range(n + 1)]
    for i in range(n):
        x[i], _, _ = proximal_step(w[i], func2, alpha)
        y, _, fy = proximal_step(2 * x[i] - w[i], func1, alpha)
        w[i + 1] = w[i] + theta * (y - x[i])

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x[0] - xs) ** 2 <= 1)

    # Set the performance metric to the final distance to the optimum in function values
    problem.set_performance_metric((func2.value(y) + fy) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    # when theta = 1
    theoretical_tau = 1 / (n + 1)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Douglas Rachford Splitting in function values ***')
        print('\tPEP-it guarantee:\t f(y_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee :\t f(y_n)-f_* <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))
    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1.

    # Test scheme parameters
    alpha = 1
    theta = 1
    n = 10

    pepit_tau, theoretical_tau = wc_drs_2(L=L,
                                          alpha=alpha,
                                          theta=theta,
                                          n=n)
