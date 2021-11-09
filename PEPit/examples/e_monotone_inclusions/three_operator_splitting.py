from PEPit.pep import PEP
from PEPit.operators.monotone import MonotoneOperator
from PEPit.operators.cocoercive import CocoerciveOperator
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_tos(L, mu, beta, alpha, theta, verbose=True):
    """
    Consider the monotone inclusion problem
        Find x, 0 \in Ax + Bx + Cx,
    where A is maximally monotone, B is cocoercive and C is the gradient of a smooth strongly convex function.
    We denote JA ad JB the respective resolvents of A and B.

    This code computes a worst-case guarantee for the Three Operator Splitting (TOS). One iteration of the algorithm
    starting from a point w is as follows:
        x = JB(w)
        y = JA( 2* x - w - Cx)
        z = w - theta * (x - y)
    and z is chosen as the next iterate.

    Given two initial points w_0 and w_1, this code computes the smallest possible tau(n,L,mu) such that the guarantee
        || (w0 - theta * (x0 - y0)) - (w1 - theta * (x1 - y1))||^2 <= tau(n,L,mu) * || w_0 - w_1||^2,
    is valid, where z_0 and z_1 are obtained after one iteration of TOS from respectively w_0 and w_1.

    :param L: (float) the Lipschitz parameter.
    :param mu: (float) the strong convexity parameter.
    :param beta: (float) the cocoercive parameter.
    :param alpha: (float) the step size in the resolvent.
    :param theta: (float) algorithm parameter.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a monotone operator
    A = problem.declare_function(MonotoneOperator, param={})
    B = problem.declare_function(CocoerciveOperator, param={'beta': beta})
    C = problem.declare_function(SmoothStronglyConvexFunction, param={'L': L, 'mu': mu})

    # Then define the starting points w0 and w1
    w0 = problem.set_initial_point()
    w1 = problem.set_initial_point()

    # Set the initial constraint that is the distance between w0 and w1
    problem.set_initial_condition((w0 - w1) ** 2 <= 1)

    # Compute one step of the Three Operator Splitting starting from w0
    x0, _, _ = proximal_step(w0, B, alpha)
    y0, _, _ = proximal_step(2 * x0 - w0 - alpha * C.gradient(x0), A, alpha)
    z0 = w0 - theta * (x0 - y0)

    # Compute one step of the Three Operator Splitting starting from w1
    x1, _, _ = proximal_step(w1, B, alpha)
    y1, _, _ = proximal_step(2 * x1 - w1 - alpha * C.gradient(x1), A, alpha)
    z1 = w1 - theta * (x1 - y1)

    # Set the performance metric to the distance between z0 and z1
    problem.set_performance_metric((z0 - z1) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = None

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Three Operator Splitting ***')
        print('\tPEP-it guarantee:\t || (w0 - theta * (x0 - y0)) - (w1 - theta * (x1 - y1))||^2 <= {:.6} ||w_1 - w_0||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    mu = 0.1
    beta = 1
    alpha = 0.9
    theta = 1.3

    pepit_tau, theoretical_tau = wc_tos(L=L,
                                        mu=mu,
                                        beta=beta,
                                        alpha=alpha,
                                        theta=theta)
