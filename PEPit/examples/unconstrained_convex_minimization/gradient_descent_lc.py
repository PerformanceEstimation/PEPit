import numpy as np
import scipy.optimize as optimize

from PEPit import PEP
from PEPit import Point

from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.operators import SymmetricLinearOperator
from PEPit.operators import SkewSymmetricLinearOperator
from PEPit.operators import LinearOperator


def wc_gradient_descent_lc(mug, Lg, typeM, muM, LM, gamma, n, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: g_\\star \\triangleq \\min_x g(Mx),

    where :math:`g` is an :math:`L_g`-smooth, `\\mu_g`-strongly convex function and :math:`M` is a general, symmetric or
    skew-symmetric matrix with :math:`\\mu_M \\leqslant \\|M\\| \\leqslant L_M`.

    Note:
        For general and skew-symmetric matrices, :math:`\\mu_M` must be set to 0.

    This code computes a worst-case guarantee for **gradient descent** with fixed step-size :math:`\\gamma`
    on a function :math:`F` defined as the composition of a function :math:`G` and a linear operator :math:`M`.
    That is, it computes the smallest possible :math:`\\tau(n, \\mu_g, L_g, \\mu_M, L_M, \\gamma)`
    such that the guarantee

    .. math:: g(Mx_n) - g_\\star \\leqslant \\tau(n, \\mu_g, L_g, \\mu_M, L_M, \\gamma) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of gradient descent run on :math:`F` with fixed step-size :math:`\\gamma`,
    where :math:`x_\\star` is a minimizer of :math:`F(x) = g(Mx)`, and where :math:`g_\\star = g(M x_\\star)`.

    In short, for given values of :math:`n`, :math:`\\mu_g`, :math:`L_g`, :math:`\\mu_M`, :math:`L_M`,
    and :math:`\\gamma`, :math:`\\tau(n, \\mu_g, L_g, \\mu_M, L_M, \\gamma)` is computed as the worst-case
    value of :math:`g(Mx_n)-g_\\star` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    Gradient descent on such a function is described by

    .. math:: x_{t+1} = x_t - \\gamma M^T \\nabla g(Mx_t),

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**:
    When :math:`\\gamma \\leqslant \\frac{2}{L}`, :math:`0 \\leqslant \\mu_g \\leqslant L_g`,
    and :math:`0 \\leqslant \\mu_M \\leqslant L_M`,
    the following **tight** theoretical guarantee is **conjectured** in [1, Conjecture 4.2],
    and the associated **lower** theoretical guarantee is stated in [1, Conjecture 4.3]:

    .. math:: g(Mx_n)-g_\\star \\leqslant \\frac{L}{2} \\max\\left\\{\\frac{\\kappa_g {M^*}^2}{\\kappa_g -1 + (1-\\kappa_g {M^*}^2 L \\gamma )^{-2n}}, (1-L\\gamma)^{2n} \\right\\} \\|x_0-x_\\star\\|^2,

    where :math:`L = L_g L_M^2`, :math:`\kappa_g = \\frac{\\mu_g}{L_g}`, :math:`\kappa_M = \\frac{\\mu_M}{L_M}`,
    :math:`M^* = \\mathrm{proj}_{[\\kappa_M,1]} \\left(\\sqrt{\\frac{h_0}{L\\gamma}}\\right)` for :math:`h_0` solution of

    .. math:: (1-\\kappa_g)(1-\kappa_g h_0)^{2n+1} = 1 - (2n+1)\\kappa_g h_0.

    **References**:

    `[1] N. Bousselmi, J. Hendrickx, F. Glineur  (2023).
    Interpolation Conditions for Linear Operators and applications to Performance Estimation Problems.
    arXiv preprint
    <https://arxiv.org/pdf/2302.08781.pdf>`_

    Args:
        mug (float): the strong convexity parameter of :math:`g(y)`.
        Lg (float): the smoothness parameter of :math:`g(y)`.
        typeM (string): type of matrix :math:`M` ("gen", "sym" or "skew").
        muM (float): lower bound on :math:`\\|M\\|` (if typeM != "sym", then muM must be set to zero).
        LM (float): upper bound on :math:`\\|M\\|`.
        gamma (float): step-size.
        n (int): number of iterations.
        verbose (int): Level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> Lg = 3.; mug = 0.3
        >>> typeM = "sym"; LM = 1.; muM = 0.
        >>> L = Lg*LM**2
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_lc(mug = mug, Lg=Lg, typeM=typeM, muM = muM, LM=LM, gamma=1 / L, n=3, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 16x16
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (2 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 3 function(s)
        			Function 1 : Adding 20 scalar constraint(s) ...
        			Function 1 : 20 scalar constraint(s) added
        			Function 2 : Adding 20 scalar constraint(s) ...
        			Function 2 : 20 scalar constraint(s) added
        			Function 2 : Adding 2 lmi constraint(s) ...
        		 Size of PSD matrix 1: 5x5
        		 Size of PSD matrix 2: 4x4
        			Function 2 : 2 lmi constraint(s) added
        			Function 3 : Adding 0 scalar constraint(s) ...
        			Function 3 : 0 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.16380641240039548
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 8.261931885644884e-09
        		All required PSD matrices are indeed positive semi-definite
        		All the primal scalar constraints are verified up to an error of 1.7219835096598865e-08
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual matrices to lmi are positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 5.282963298309745e-07
        (PEPit) Final upper bound (dual): 0.1638061803039686 and lower bound (primal example): 0.16380641240039548 
        (PEPit) Duality gap: absolute: -2.320964268831549e-07 and relative: -1.4168946348439444e-06
        *** Example file: worst-case performance of gradient descent on g(Mx) with fixed step-sizes ***
        	PEPit guarantee:	 f(x_n)-f_* <= 0.163806 ||x_0 - x_*||^2
        	Theoretical guarantee:	 f(x_n)-f_* <= 0.16379 ||x_0 - x_*||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function G and a linear operator M
    G = problem.declare_function(SmoothStronglyConvexFunction, mu=mug, L=Lg)
    if typeM == "gen":
        M = problem.declare_function(function_class=LinearOperator, L=LM)
    elif typeM == "sym":
        M = problem.declare_function(function_class=SymmetricLinearOperator, mu=muM, L=LM)
    elif typeM == "skew":
        M = problem.declare_function(function_class=SkewSymmetricLinearOperator, L=LM)
    else:
        raise ValueError("The argument \'typeM\' must be \'gen\', \`sym\` or \`skew\`."
                         "Got {}".format(typeM))

    # Define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Defining unique optimal point xs = x_* of F(x) = g(Mx) and corresponding function value fs = f_*
    xs = Point()                          # xs
    ys = M.gradient(xs)                   # ys = M*xs
    us, fs = G.oracle(ys)                 # us = \nabla g(ys) and fs = F(xs) = g(M*ys)
    if typeM == "gen":
        vs = M.T.gradient(us)             # vs =  M^T \nabla g(ys) = \nabla F(xs)
    elif typeM == "sym":
        vs = M.gradient(us)               # vs =  M \nabla g(ys) = \nabla F(xs)
    elif typeM == "skew":
        vs = -M.gradient(us)              # vs =  -M \nabla g(ys) = \nabla F(xs)
    else:
        raise ValueError("The argument \'typeM\' must be \'gen\', \`sym\` or \`skew\`."
                         "Got {}".format(typeM))
    problem.add_constraint(vs ** 2 == 0)  # vs = \nabla F(xs) = 0

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the GD method on F
    x = x0
    for _ in range(n):
        y = M.gradient(x)        # y = Mx
        u = G.gradient(y)        # y = \nabla g(y)
        if typeM == "gen":
            v = M.T.gradient(u)  # v = M^T u
        elif typeM == "sym":
            v = M.gradient(u)    # v = M u
        elif typeM == "skew":
            v = -M.gradient(u)   # v = M^T u = - M u
        else:
            raise ValueError("The argument \'typeM\' must be \'gen\', \`sym\` or \`skew\`."
                             "Got {}".format(typeM))
        x = x - gamma * v

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric(G(M.gradient(x)) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    L = Lg * LM ** 2
    kappag = mug / Lg
    kappaM = muM / LM

    def fun(x):
        return (1 - (2 * n + 1) * x) * (1 - x) ** (-2 * n - 1) - 1 + kappag

    x = optimize.fsolve(fun, 0.5, xtol=1e-10, full_output=True)[0]
    h0 = x / kappag
    t = np.sqrt(h0[0] / (L * gamma))

    if t < kappaM:
        M_star = kappaM
    elif t > 1:
        M_star = 1
    else:
        M_star = t

    theoretical_tau = 0.5 * L * np.max((kappag * M_star ** 2 / (
                kappag - 1 + (1 - kappag * L * gamma * M_star ** 2) ** (-2 * n)), (1 - L * gamma) ** (2 * n)))

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent on g(Mx) with fixed step-sizes ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    Lg = 3
    mug = 0.3
    typeM = "gen"
    LM = 1.
    muM = 0.1

    pepit_tau, theoretical_tau = wc_gradient_descent_lc(mug=mug, Lg=Lg, typeM=typeM, muM=muM, LM=LM,
                                                        gamma=1 / (Lg * LM ** 2), n=3, verbose=1)
