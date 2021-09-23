import cvxpy as cp

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_convex_function import SmoothConvexFunction
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_drs_2(L, alpha, theta, n):
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

     We show how to compute the worst-case value of F(yN)-F(xs) when yN is
    obtained by doing N steps of (relaxed) DRS starting with an initial
    iterate w0 satisfying ||x0-xs||<=1.

    It is known that xk and yk converge to xs, but not wk, and hence
    we require the initial condition on x0 (arbitrary choice; partially
    justified by the fact we choose f2 to be the smooth function).
    Note that yN is feasible as it has a finite value for f1
    (output of the proximal opertor on f1) and as f2 is smooth

    :param L: (float) the smoothness parameter.
    :param alpha: (float) parameter of the scheme.
    :param theta: (float) parameter of the scheme.
    :param n: (int) number of iterations.
    :return:
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex lipschitz function
    func1 = problem.declare_function(SmoothConvexFunction,
                                    {'L': L})
    func2 = problem.declare_function(ConvexFunction,
                                     {})
    func = func1 + func2

    # Start by defining its unique optimal point and its function value
    xs = func.optimal_point()
    fs = func.value(xs)
    fs1 = func1.value(xs)
    fs2 = func2.value(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()
    _ = func1.value(x0)
    _ = func2.value(x0)
    _ = func.value(x0)

    # Compute trajectory starting from x0
    x = [x0 for _ in range(n)]
    w = [x0 for _ in range(n+1)]
    for i in range(n):
        x[i], _, _ = proximal_step(w[i], func2, alpha)
        y, _, fy = proximal_step(2 * x[i] - w[i], func1, alpha)
        w[i+1] = w[i] + theta * (y-x[i])

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric((func2.value(y) + fy) - (fs1 + fs2))

    # Solve the PEP
    wc = problem.solve(cp.MOSEK)

    # Theoretical guarantee (for comparison)
    # when theta = 1
    theory = 1/n
    print('*** Example file: worst-case performance of the Douglas Rachford Splitting in function values ***')
    print('\tPEP-it guarantee:\tf(y_n) - f_* <= ', wc)
    print('\tTheoretical guarantee when theta = 1 :\tf(y_n) - f_* <=  <= ', theory)
    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)

    # Return the rate of the evaluated method
    return wc


if __name__ == "__main__":

    L = 1.
    ## Test scheme parameters
    alpha = 1
    theta = 1
    n = 5

    rate = wc_drs_2(L=L,
                    alpha=alpha,
                    theta=theta,
                    n=n)