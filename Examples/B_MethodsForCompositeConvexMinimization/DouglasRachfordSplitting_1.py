from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.proximal_step import proximal_step

def wc_drs(mu, L, alpha, theta, n):
    """
    In this example, we use a Douglas-Rachford splitting (DRS)
    method for solving the composite convex minimization problem
    min_x { F(x) = f_1(x) + f_2(x) }
    (for notational convenience we denote xs=argmin_x F(x);
    where f_1(x) is L-smooth and mu-strongly convex, and f_2 is convex,
    closed and proper. Both proximal operators are assumed to be available.

    We show how to compute a contraction factor for the iterates of DRS
    (i.e., how do the iterates contract when the algorithm is started from
    two different initial points).

    Our notations for the DRS algorithm are as follows:
        x_k     = prox_{\alpha f2}(w_k)
        y_k     = prox_{\alpha f1}(2*x_k-w_k)
        w_{k+1} = w_k +\theta (y_k - x_k)
    and our goal is to compute the smallest contraction factor rho such that
    when the algorithm is started from two different points w_0 and w_0', we
    have ||w_1 - w_1'||^2 <= rho^2 ||w_0 - w_0'||^2.

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
    :return:
    """


    # Instantiate PEP
    problem = PEP()

    # Declare a convex lipschitz function
    func1 = problem.declare_function(SmoothStronglyConvexFunction,
                                    {'mu': mu, 'L': L})
    func2 = problem.declare_function(ConvexFunction, {})
    func = func1 + func2

    # Start by defining its unique optimal point and its function value
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()
    x0p = problem.set_initial_point()
    f0 = func.value(x0)

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - x0p)**2 <= 1)

    # Compute trajectory starting from x0
    w = x0
    for _ in range(n):
        x, _, _ = proximal_step(w, func2, alpha)
        y, _, _ = proximal_step(2 * x - w, func1, alpha)
        w = w + theta * (y-x)

    # Compute trajectory starting from x0p
    wp = x0p
    for _ in range(n):
        xp, _, _ = proximal_step(wp, func2, alpha)
        yp, _, _ = proximal_step(2 * xp - wp, func1, alpha)
        wp = wp + theta * (yp - xp)

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric((w - wp)**2)

    # Solve the PEP
    wc = problem.solve()

    # Theoretical guarantee (for comparison)
    # when theta = 1
    theory = (max(1/(1+mu*alpha), alpha*1/(1+alpha*L)))**(2*n)
    print('*** Example file: worst-case performance of the Douglas Rachford Splitting in distance ***')
    print('\tPEP-it guarantee:\t||w^2_n - w^1_n||^2 <= ', wc)
    print('\tTheoretical guarantee :\t|w^2_n - w^1_n||^2 <= ', theory)
    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)

    # Return the rate of the evaluated method
    return wc


if __name__ == "__main__":

    mu = 0.1
    L = 1.
    ## Test scheme parameters
    alpha = 3
    theta = 1
    n = 2

    rate = wc_drs(mu=mu,
                    L=L,
                    alpha=alpha,
                    theta=theta,
                    n=n)