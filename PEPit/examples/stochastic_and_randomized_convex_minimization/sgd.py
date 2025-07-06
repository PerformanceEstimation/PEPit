import numpy as np

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def wc_sgd(L, mu, gamma, v, R, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the finite sum minimization problem

    .. math:: F_\\star \\triangleq \\min_x \\left\\{F(x) \\equiv \\frac{1}{n} \\sum_{i=1}^n f_i(x)\\right\\},

    where :math:`f_1, ..., f_n` are :math:`L`-smooth and :math:`\\mu`-strongly convex.
    In the sequel, we use the notation :math:`\\mathbb{E}` for denoting the expectation over the uniform distribution
    of the index :math:`i \\sim \\mathcal{U}\\left([|1, n|]\\right)`,
    e.g., :math:`F(x)\\equiv\\mathbb{E}[f_i(x)]`. In addition, we assume a bounded variance at
    the optimal point (which is denoted by :math:`x_\\star`):

    .. math:: \\mathbb{E}\\left[\\|\\nabla f_i(x_\\star)\\|^2\\right] = \\frac{1}{n} \\sum_{i=1}^n\\|\\nabla f_i(x_\\star)\\|^2 \\leqslant v^2.

    This code computes a worst-case guarantee for one step of the **stochastic gradient descent** (SGD) in expectation,
    for the distance to an optimal point. That is, it computes the smallest possible
    :math:`\\tau(L, \\mu, \\gamma, v, R, n)` such that

    .. math:: \\mathbb{E}\\left[\\|x_1 - x_\\star\\|^2\\right] \\leqslant \\tau(L, \\mu, \\gamma, v, R, n)

    where :math:`\\|x_0 - x_\\star\\|^2 \\leqslant R^2`, where :math:`v` is the variance at :math:`x_\\star`, and where
    :math:`x_1` is the output of one step of SGD (note that we use the notation :math:`x_0,x_1` to denote two
    consecutive iterates for convenience; as the bound is valid for all :math:`x_0`, it is also valid for
    any pair of consecutive iterates of the algorithm).

    **Algorithm**: One iteration of SGD is described by:

    .. math::
        \\begin{eqnarray}
            \\text{Pick random }i & \\sim & \\mathcal{U}\\left([|1, n|]\\right), \\\\
            x_{t+1} & = & x_t - \\gamma \\nabla f_{i}(x_t),
        \\end{eqnarray}

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**: An empirically tight one-iteration guarantee is provided in the code of PESTO [1]:

        .. math:: \\mathbb{E}\\left[\\|x_1 - x_\\star\\|^2\\right] \\leqslant \\left( \\gamma\\frac{L-\\mu}{2}R + \\sqrt{\\left(1-\\gamma\\frac{L+\\mu}{2}\\right)^2 R^2 + \\gamma^2 v^2} \\right)^2.

    Note that we observe the guarantee does not depend on the number :math:`n` of
    functions for this particular setting, thereby implying that the guarantees are also valid for expectation
    minimization settings (i.e., when :math:`n` goes to infinity).

    **References**: Empirically tight guarantee provided in code of [1]. Using SDPs for analyzing SGD-type method was
    proposed in [2, 3].

    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017). Performance Estimation Toolbox (PESTO): automated worst-case
    analysis of first-order optimization methods. In 56th IEEE Conference on Decision and Control (CDC).
    <https://github.com/AdrienTaylor/Performance-Estimation-Toolbox>`_

    `[2] B. Hu, P. Seiler, L. Lessard (2020). Analysis of biased stochastic gradient descent using sequential
    semidefinite programs. Mathematical programming.
    <https://arxiv.org/pdf/1711.00987.pdf>`_

    `[3] A. Taylor, F. Bach (2019). Stochastic first-order methods: non-asymptotic and computer-aided analyses
    via potential functions. Conference on Learning Theory (COLT).
    <https://arxiv.org/pdf/1902.00947.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        gamma (float): the step-size.
        v (float): the variance bound.
        R (float): the initial distance.
        n (int): number of functions.
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): level of information details to print.

                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + solver details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> mu = 0.1
        >>> L = 1
        >>> gamma = .7 / L
        >>> pepit_tau, theoretical_tau = wc_sgd(L=L, mu=mu, gamma=gamma, v=1, R=2, n=5, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 11x11
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (2 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 5 function(s)
                    Function 1 : Adding 2 scalar constraint(s) ...
                    Function 1 : 2 scalar constraint(s) added
                    Function 2 : Adding 2 scalar constraint(s) ...
                    Function 2 : 2 scalar constraint(s) added
                    Function 3 : Adding 2 scalar constraint(s) ...
                    Function 3 : 2 scalar constraint(s) added
                    Function 4 : Adding 2 scalar constraint(s) ...
                    Function 4 : 2 scalar constraint(s) added
                    Function 5 : Adding 2 scalar constraint(s) ...
                    Function 5 : 2 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 4.1830003941347
        (PEPit) Primal feasibility check:
                The solver found a Gram matrix that is positive semi-definite
                All the primal scalar constraints are verified up to an error of 1.0321604682062002e-15
        (PEPit) Dual feasibility check:
                The solver found a residual matrix that is positive semi-definite
                All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 1.3303453802862415e-07
        (PEPit) Final upper bound (dual): 4.183000393650909 and lower bound (primal example): 4.1830003941347
        (PEPit) Duality gap: absolute: -4.837907852106582e-10 and relative: -1.1565640440508152e-10
        *** Example file: worst-case performance of stochastic gradient descent with fixed step-size ***
            PEPit guarantee:	 E[||x_1 - x_*||^2] <= 4.183
            Theoretical guarantee:	 E[||x_1 - x_*||^2] <= 4.183

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    fn = [problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu) for _ in range(n)]
    func = np.mean(fn)

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the bounded variance and the distance between initial point and optimal one
    var = np.mean([f.gradient(xs) ** 2 for f in fn])

    problem.add_constraint(var <= v ** 2)
    problem.set_initial_condition((x0 - xs) ** 2 <= R ** 2)

    # Compute the *expected* distance to optimality after running one step of the stochastic gradient descent
    distavg = np.mean([(x0 - gamma * f.gradient(x0) - xs) ** 2 for f in fn])

    # Set the performance metric to the distance average to optimal point
    problem.set_performance_metric(distavg)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (gamma * R * (L - mu) / 2 + np.sqrt(
        R ** 2 * (L + mu) ** 2 / 4 * (gamma - 2 / (L + mu)) ** 2 + v ** 2 * gamma ** 2)) ** 2

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of stochastic gradient descent with fixed step-size ***')
        print('\tPEPit guarantee:\t E[||x_1 - x_*||^2] <= {:.6}'.format(pepit_tau))
        print('\tTheoretical guarantee:\t E[||x_1 - x_*||^2] <= {:.6}'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    mu = 0.1
    L = 1
    gamma = .7 / L
    pepit_tau, theoretical_tau = wc_sgd(L=L, mu=mu, gamma=gamma, v=1, R=2, n=5, wrapper="cvxpy", solver=None, verbose=1)
