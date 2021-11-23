import numpy as np

from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.functions.convex_function import ConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_saga(L, mu, n, verbose=True):
    """
    Consider the finite sum minimization problem
        f_* = min_x {F(x) = 1/n [f1(x) + ... + fn(x)] + h(x)},
    where f1, ..., fn are assumed L-smooth and mu-strongly convex, and h
    is closed proper and convex with a proximal operator available.

    This code computes the exact rate for the Lyapunov function from the original SAGA paper,
    given in Theorem 1 of [1]. At each iteration k, for a j chosen uniformely at random,
        phi_j^{k+1} = x^k
        w^{k+1} = x^k - gamma(f_j'(phi_j^{k+1}) - f_j'(phi_j^k) + 1/n sum(f_i'(phi^k)))
        x^{k+1} = prox_gamma^h(w^{k+1})

    That is, it computes the smallest possible tau(n,L,mu) such that a given Lyapunov sequence V1 is
    decreasing along the trajectory:
        V1(x_1) <= tau(n,L,mu) V0(x_0),
    with Vk(x1) = 1/n*sum(fi(phi_i^k) - f_*) - 1/n*sum(<f_i'(x^*), phi_i^k - x^*> + ||x_k - x^*||/(2*gamma*(1-mu*gamma)),
    and with gamma = 1/2/(mu*n+L).

    [1] Aaron Defazio, Francis Bach, and Simon Lacoste-Julien.
        "SAGA: A fast incremental gradient method with support for
        non-strongly convex composite objectives." (2014)
        (Theorem 1 of [1])

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param n: (int) number of functions.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function and n smooth strongly convex ones
    h = problem.declare_function(ConvexFunction,
                                 param={}, is_differentiable=True)
    fn = [problem.declare_function(SmoothStronglyConvexFunction,
                                   param={'L': L, 'mu': mu}, is_differentiable=True) for _ in range(n)]

    # Define the objective as a linear combination of the former
    func = h + np.mean(fn)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()

    # Then define the initial points
    phi = [problem.set_initial_point() for _ in range(n)]
    x0 = problem.set_initial_point()

    # Compute the initial value of the Lyapunov function, for a given parameter
    gamma = 1 / 2 / (mu * n + L)
    c = 1 / 2 / gamma / (1 - mu * gamma) / n
    init_lyapunov = c * (xs - x0) ** 2
    for i in range(n):
        gi, fi = fn[i].oracle(phi[i])
        gis, fis = fn[i].oracle(xs)
        init_lyapunov = init_lyapunov + 1 / n * (fi - fis - gis * (phi[i] - xs))

    # Set the initial constraint as the Lyapunov bounded by 1
    problem.set_initial_condition(init_lyapunov <= 1)

    # Compute the expected value of the Lyapunov function after one iteration
    # (so: expectation over n possible scenarios: one for each element fi in the function).
    final_lyapunov_avg = (xs - xs) ** 2
    for i in range(n):
        w = x0 - gamma * (fn[i].gradient(x0) - fn[i].gradient(phi[i]))
        for j in range(n):
            w = w - gamma/n * fn[j].gradient(phi[j])
        x1, _, _ = proximal_step(w, h, gamma)
        final_lyapunov = c * (x1 - xs) ** 2
        for j in range(n):
            gis, fis = fn[j].oracle(xs)
            if i != j:
                gi, fi = fn[j].oracle(phi[j])
                final_lyapunov = final_lyapunov + 1 / n * (fi - fis - gis * (phi[j] - xs))
            else:
                gi, fi = fn[j].oracle(x0)
                final_lyapunov = final_lyapunov + 1 / n * (fi - fis - gis * (x0 - xs))
        final_lyapunov_avg = final_lyapunov_avg + final_lyapunov / n

    # Set the performance metric to the distance average to optimal point
    problem.set_performance_metric(final_lyapunov_avg)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison) : the bound is given in theorem 1 of [1]
    kappa = 1 / gamma / mu
    theoretical_tau = (1 - 1 / kappa)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of SAGA for a given Lyapunov function ***')
        print('\tPEP-it guarantee:\t\t v1(x_1, x_*) <= {:.6} v0(x_0, x_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t v1(x_1, x_*) <= {:.6} vO(x_0, x_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 5
    L = 1
    mu = 0.1

    pepit_tau, theoretical_tau = wc_saga(L=L,
                                         mu=mu,
                                         n=n)
