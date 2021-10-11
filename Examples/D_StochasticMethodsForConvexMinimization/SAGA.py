import cvxpy as cp
import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.proximal_step import proximal_step



def wc_saga(L, mu, n, verbose=True):
    """
    Consider the finite sum minimization problem
        f_* = min_x F(x) = 1/n [f1(x) + ... + fn(x)] + h(x),
    where f1, ..., fn are assumed L-smooth and mu-strongly convex, and h
    is closed proper and convex with a proximal operator available.

    This code computes the exact rate for the Lyapunov function from the original SAGA paper,
     given in Theorem 1 of [1].

    [1] Aaron Defazio, Francis Bach, and Simon Lacoste-Julien.
        "SAGA: A fast incremental gradient method with support for
        non-strongly convex composite objectives." (2014)
        (Theorem 1 of [1])

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function
    h = problem.declare_function(ConvexFunction,
                                 {})
    func = h
    fn = []
    for i in range(n):
        fn.append(problem.declare_function(SmoothStronglyConvexFunction,
                                           {'L': L, 'mu': mu}))
        func += fn[i]/n

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the initial values
    phi = []
    for i in range(n):
        phi.append(problem.set_initial_point())
    x0 = problem.set_initial_point()

    # Compute the initial value of the Lyapunov function, for a given parameter
    gamma = 1/2/(mu*n + L)
    c = 1/2/gamma/(1 - mu*gamma)/n
    T0 = c * (xs - x0)**2
    for i in range(n):
        gi, fi = fn[i].oracle(phi[i])
        gis, fis = fn[i].oracle(xs)
        T0 = T0 + 1/n * (fi - fis - gis * (phi[i] - xs))

    # Set the initial constraint as the Lyapunov bounded by 1
    problem.set_initial_condition(T0 <= 1.)

    # Compute the expected value of te Lyapunov function after one iteration
    #(so: expectation over n possible scenarios:  one for each element fi in the function).

    T1avg = (xs - xs)**2
    for i in range(n):
        w = x0 - gamma * (fn[i].gradient(x0) - fn[i].gradient(phi[i]))
        for j in range(n):
            w = w - gamma/n * fn[j].gradient(phi[j])
        x1, _, _ = proximal_step(w, h, gamma)
        T1 = c * (xs - x1) ** 2
        for j in range(n):
            gis, fis = fn[j].oracle(xs)
            if i != j:
                gi, fi = fn[j].oracle(phi[j])
                T1 = T1 + 1/n * (fi - fis - gis * (phi[j] - xs))
            else:
                gi, fi = fn[j].oracle(x0)
                T1 = T1 + 1/n * (fi - fis - gis * (x0 - xs))
        T1avg = T1avg + T1/n

    # Set the performance metric to the distance average to optimal point
    problem.set_performance_metric(T1avg)

    # Solve the PEP
    pepit_tau = problem.solve(solver=cp.MOSEK, verbose=verbose)

    # Compute theoretical guarantee (for comparison) : the bound is given in theorem 1 of [1]
    kappa = 1/gamma/mu
    theoretical_tau = (1 - 1/kappa)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of SAGA for a given Lyapunov function ***')
        print('\tPEP-it guarantee:\t\t T1(x_n, x_*) <= {:.6} TO(x_n, x_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t T1(x_n, x_*) <= {:.6} TO(x_n, x_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 5
    L = 1
    mu = 0.1

    pepit_tau, theoretical_tau = wc_saga(L=L,
                                        mu=mu,
                                        n=n)