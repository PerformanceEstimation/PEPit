from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.functions.convex_function import ConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_adrs(mu, L, alpha, n, verbose=True):
    """
    Consider the composite convex minimization problem,
        min_x { F(x) = f_1(x) + f_2(x) }
    where f_1 is closed convex and proper, and f_2 is L-smooth mu-strongly convex.

    This code computes a worst-case guarantee for the fast Douglas Rachford Splitting method:
           x_k     = prox_{\alpha f2}(u_k)
           y_k     = prox_{\alpha f1}(2*x_k-u_k)
           w_{k+1} = u_k + \theta (y_k - x_k)
           if k > 1
               u{k+1} = w{k+1} + (k-2)/(k+1) * (w{k+1} - w{k});
           else
               u{k+1} = w{k+1};

    That is, it computes the smallest possible tau(n,L,mu,alpha) such that the guarantee
        F(y_n) - F(x_*) <= tau(n,L,mu,alpha) * ||w_0 - w_*||^2
    is valid, where x_n is the output of the Fast Douglas Rachford Splitting method, and where x_* is a minimizer of F,
    and w_* defined such that
        x_* = prox_{\alpha}(w_*) is an optimal point.

    The detailed approach is available in
    [1] Panagiotis Patrinos, Lorenzo Stella, and Alberto Bemporad.
        "Douglas-Rachford splitting: Complexity estimates and accelerated
        variants." In 53rd IEEE Conference on Decision and Control (2014)
        where the theory is available for quadratics.

    The tight guarantee obtained in [1, Theorem 4] is :
        tau_q(n,L,mu,alpha) = 2/alpha*(1+alpha*L)/(1-alpha*L)/(n+2)**2.
    However, this guarantee is only valid for quadratics. So we expect :
        tau_q(n,L,mu,alpha) <= tau(n,L,mu,alpha).

    :param mu: (float) the strong convexity parameter.
    :param L: (float) the smoothness parameter.
    :param alpha: (float) the parameter of the scheme.
    :param n: (int) the number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function and a smooth strongly convex function
    func1 = problem.declare_function(ConvexFunction, param={})
    func2 = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    fs = func.value(xs)
    g1s, _ = func1.oracle(xs)
    g2s, _ = func2.oracle(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()
    f0 = func.value(x0)

    # Set the parameters of the scheme
    theta = (1 - alpha * L) / (1 + alpha * L)

    # Set the initial constraint that is the distance between x0 and ws = w^*
    ws = xs + alpha * g2s
    problem.set_initial_condition((ws - x0) ** 2 <= 1)

    # Compute n steps of the Fast Douglas Rachford Splitting starting from x0
    x = [x0 for _ in range(n)]
    w = [x0 for _ in range(n + 1)]
    u = [x0 for _ in range(n + 1)]
    for i in range(n):
        x[i], _, _ = proximal_step(u[i], func2, alpha)
        y, _, fy = proximal_step(2 * x[i] - u[i], func1, alpha)
        w[i + 1] = u[i] + theta * (y - x[i])
        if i >= 1:
            u[i + 1] = w[i + 1] + (i - 1) / (i + 2) * (w[i + 1] - w[i])
        else:
            u[i + 1] = w[i + 1]

    # Set the performance metric to the final distance in function values to optimum
    problem.set_performance_metric(func2.value(y) + fy - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 2 / (alpha * theta * (n + 3) ** 2)

    # Print conclusion if required
    if verbose:
        print('*** Example file:'
              ' worst-case performance of the Accelerated Douglas Rachford Splitting in function values ***')
        print('\tPEP-it guarantee:\t f(y_n)-f_* <= {:.6} ||x0 - ws||^2'.format(pepit_tau))
        print('\tTheoretical guarantee for quadratics :\t f(y_n)-f_* <= {:.6} ||x0 - ws||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    mu = 0.1
    L = 1.

    # Test scheme parameters
    alpha = 0.9
    n = 2

    pepit_tau, theoretical_tau = wc_adrs(mu=mu,
                                         L=L,
                                         alpha=alpha,
                                         n=n)
