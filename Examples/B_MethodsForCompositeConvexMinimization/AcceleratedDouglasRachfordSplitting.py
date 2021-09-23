import cvxpy as cp

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_adrs(mu, L, alpha, n):
    """
    In this example, we use a fast Douglas-Rachford splitting
    method for solving the composite convex minimization problem
        min_x { F(x) = f_1(x) + f_2(x) }
    (for notational convenience we denote xs=argmin_x F(x);
    where f_2 is L-smooth and f_1 is closed, proper and convex.

    The method below is due to
    [1] Panagiotis Patrinos, Lorenzo Stella, and Alberto Bemporad.
         "Douglas-Rachford splitting: Complexity estimates and accelerated
         variants." In 53rd IEEE Conference on Decision and Control (2014)
    where the theory is available for quadratics.

    Our notations for the algorithms are as follows:
           x_k     = prox_{\alpha f2}(u_k)
           y_k     = prox_{\alpha f1}(2*x_k-u_k)
           w_{k+1} = u_k + \theta (y_k - x_k)
           if k > 1
               u{k+1} = w{k+1} + (k-2)/(k+1) * (w{k+1} - w{k});
           else
               u{k+1} = w{k+1};
           end

    In Douglas-Rachford schemes, w_{k} converge to a point ws such that
           xs = prox_{\alpha}(ws) is an optimal point.
    Hence we show how to compute the worst-case behavior of F(y{N})-F(xs)
    given that ||w0 - ws || <= 1.

    :param mu: (float) the strong convexity parameter.
    :param L: (float) the smoothness parameter.
    :param alpha: (float) parameter of the scheme.
    :param n: (int) number of iterations.
    :return:
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex lipschitz function
    func1 = problem.declare_function(SmoothStronglyConvexFunction,
                                    {'mu': mu, 'L': L})
    func2 = problem.declare_function(ConvexFunction,
                                     {})
    func = func1 + func2

    # Start by defining its unique optimal point and its function value
    xs = func.optimal_point()
    fs = func.value(xs)
    g1, f1 = func1.oracle(xs)
    g2, f2 = func2.oracle(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()
    _ = func1.value(x0)
    _ = func2.value(x0)

    # Parameters of the scheme
    theta = (1-alpha*L)/(1+alpha*L)

    # Set the initial constraint that is the distance between x0 and x^*
    ws = xs + alpha * g2
    problem.set_initial_condition((ws - x0) ** 2 <= 1)

    # Compute trajectory starting from x0
    x = [x0 for _ in range(n)]
    w = [x0 for _ in range(n+1)]
    u = [x0 for _ in range(n+1)]
    for i in range(n):
        x[i], _, _ = proximal_step(u[i], func2, alpha)
        y, _, fy = proximal_step(2 * x[i] - u[i], func1, alpha)
        w[i+1] = u[i] + theta * (y-x[i])
        if i >= 1:
            u[i+1] = w[i+1] + (i-1)/(i+2) * (w[i+1] - w[i])
        else:
            u[i+1] = w[i+1]

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric((func2.value(y) + fy) - fs)

    # Solve the PEP
    wc = problem.solve(solver=cp.MOSEK)

    # Theoretical guarantee (for comparison)
    # when theta = 1
    theory = 2/(alpha*theta*(n+2)**2)
    print('*** Example file: worst-case performance of the Accelerated Douglas Rachford Splitting in function values ***')
    print('\tPEP-it guarantee:\tf(y_n) - f_* <= ', wc)
    print('\tTheoretical guarantee on quadratics = 1 :\tf(y_n) - f_* <=  <= ', theory)
    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)

    # Return the rate of the evaluated method
    return wc


if __name__ == "__main__":

    mu = 0.1
    L = 1.
    ## Test scheme parameters
    alpha = 0.9
    n = 2

    rate = wc_adrs(mu=mu,
                   L=L,
                   alpha=alpha,
                   n=n)