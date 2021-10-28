from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_drs(mu, L, alpha, theta, n, verbose=True):
    """
    Consider the composite convex minimization problem,
        min_x { F(x) = f_1(x) + f_2(x) }
    where f_1(x) is L-smooth and mu-strongly convex, and f_2 is convex,
    closed and proper. Both proximal operators are assumed to be available.

    This code computes a worst-case guarantee for the Douglas Rachford Splitting (DRS) method, where
    our notations for the DRS algorithm are as follows:
        x_k     = prox_{\alpha f2}(w_k)
        y_k     = prox_{\alpha f1}(2*x_k-w_k)
        w_{k+1} = w_k +\theta (y_k - x_k)

    That is, it computes the smallest possible tau(n,L) such that the guarantee
        ||w_1 - w_1'||^2 <= tau(n,L) * ||w_0 - w_0'||^2.
    is valid, where x_n is the output of the Fast Douglas Rachford Splitting method. It is a contraction
    factor computed when the algorithm is started from two different points w_0 and w_0'.

    Details on the SDP formulations can be found in
    [1] Ernest K. Ryu, Adrien B. Taylor, Carolina Bergeling,
      and Pontus Giselsson. "Operator splitting performance estimation:
      Tight contraction factors and optimal parameter selection." (2018)

    When theta = 1, the bound can be compared with that of
    [2] Giselsson, Pontus, and Stephen Boyd. "Linear convergence and
       metric selection in Douglas-Rachford splitting and ADMM."
       IEEE Transactions on Automatic Control (2016).

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

    # Declare a convex and a smooth strongly convex function.
    func1 = problem.declare_function(SmoothStronglyConvexFunction,
                                     {'mu': mu, 'L': L})
    func2 = problem.declare_function(ConvexFunction, {})

    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting points x0 and x0p of the algorithm
    w0 = problem.set_initial_point()
    w0p = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x0p
    problem.set_initial_condition((w0 - w0p) ** 2 <= 1)

    # Compute n steps of the Douglas Rachford Splitting starting from x0
    w = w0
    for _ in range(n):
        x, _, _ = proximal_step(w, func2, alpha)
        y, _, _ = proximal_step(2 * x - w, func1, alpha)
        w = w + theta * (y - x)

    # Compute n steps of the Douglas Rachford Splitting starting from x0p
    wp = w0p
    for _ in range(n):
        xp, _, _ = proximal_step(wp, func2, alpha)
        yp, _, _ = proximal_step(2 * xp - wp, func1, alpha)
        wp = wp + theta * (yp - xp)

    # Set the performance metric to the final distance between wp and w
    problem.set_performance_metric((w - wp) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve()

    # Compute theoretical guarantee (for comparison)
    # when theta = 1
    theoretical_tau = (max(1 / (1 + mu * alpha), alpha * 1 / (1 + alpha * L))) ** (2 * n)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Douglas Rachford Splitting in distance ***')
        print('\tPEP-it guarantee:\t ||w - wp||^2 <= {:.6} ||w0 - w0p||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t||w - wp||^2 <= {:.6} ||w0 - w0p||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    mu = 0.1
    L = 1.
    # Test scheme parameters
    alpha = 3
    theta = 1
    n = 2

    pepit_tau, theoretical_tau = wc_drs(mu=mu,
                  L=L,
                  alpha=alpha,
                  theta=theta,
                  n=n)
