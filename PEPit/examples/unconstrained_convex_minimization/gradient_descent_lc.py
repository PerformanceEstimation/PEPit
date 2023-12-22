from PEPit import PEP
from PEPit import Point
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.operators import SymmetricLinearOperator
from PEPit.operators import SkewSymmetricLinearOperator
from PEPit.operators import LinearOperator
import numpy as np
import scipy.optimize as optimize

def wc_gradient_descent_lc(mug, Lg, typeM, muM, LM, gamma, n, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: g_\\star \\triangleq \\min_x g(Mx),

    where :math:`g` is :math:`L_g`-smooth, `\\mu_g`-strongly convex and :math:`M` is a not necessarly symmetric, symmetric or
    skew-symmetric matrix with :math:`\\mu_M \\leqslant \\|M\\| \\leqslant L_M` (note that for not necessarly symmetric and
    skew-symmetric matrices, :math:`\\mu_M` must be set to 0.

    This code computes a worst-case guarantee for **gradient descent** with fixed step-size :math:`\\gamma`.
    That is, it computes the smallest possible :math:`\\tau(n, \\mu, L, \\gamma)` such that the guarantee

    .. math:: g(Mx_n) - g_\\star \\leqslant \\tau(n, \\mu, L, \\gamma) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of gradient descent with fixed step-size :math:`\\gamma`, and
    where :math:`x_\\star` is a minimizer of :math:`F(x) = g(Mx)`.

    In short, for given values of :math:`n`, :math:`\\mu`, :math:`L`, and :math:`\\gamma`, :math:`\\tau(n, \\mu, L, \\gamma)` is computed as the worst-case
    value of :math:`g(Mx_n)-g_\\star` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma M^T \\nabla g(Mx_t),

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**:
    When :math:`\\gamma \\leqslant \\frac{2}{L}`, :math:`0 \\leqslant \\mu_g \\leqslant L_g`, and :math:`0 \\leqslant \\mu_M \\leqslant L_M`
    the **tight** theoretical guarantee is **conjectured** in [1, Conjecture 4.2]:
    .. math:: g(Mx_n)-g_\\star \\leqslant \\frac{L}{2} \\max\\{\\frac{\\kappa_g {M^*}^2}{\\kappa_g -1 + (1-\\kappa_g {M^*}^2 L \\gamma )^{-2n}}, (1-L\\gamma)^{2n} \\} \\|x_0-x_\\star\\|^2,

    where :math:`L = L_g L_M^2`, :math:`\kappa_g = \\frac{\\mu_g}{L_g}`, :math:`\kappa_M = \\frac{\\mu_M}{L_M}`, :math:`M^* = \\mathrm{proj}_{[\\kappa_M,1]} (\\sqrt{\\frac{h_0}{L\\gamma}})` for :math:`h_0` solution of
    .. math:: (1-\\kappa_g)(1-\kappa_g h_0)^{2n+1} = 1 - (2n+1)\\kappa_g h_0.

    **References**:

    `[1] N. Bousselmi, J. Hendrickx, F. Glineur  (2023).
    Interpolation Conditions for Linear Operators and applications to Performance Estimation Problems.
    arXiv preprint
    <https://arxiv.org/pdf/2302.08781.pdf>`_

    Args:
        mug (float): the strong convexity parameter of :math:`g(y)`.
        Lg (float): the smoothness parameter of :math:`g(y)`.
        typeM (string): type of matrix M ("gen", "sym" or "skew").
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
        (PEPit) Setting up the problem: size of the main PSD matrix: 16x16
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (2 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
                       function 1 : Adding 20 scalar constraint(s) ...
                		 function 1 : 20 scalar constraint(s) added
                		 function 2 : Adding 72 scalar constraint(s) ...
                		 function 2 : 72 scalar constraint(s) added
                		 function 2 : Adding 1 lmi constraint(s) ...
                		 Size of PSD matrix 1: 9x9
                		 function 2 : 1 lmi constraint(s) added
                (PEPit) Setting up the problem: constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.16381772971844077
        *** Example file: worst-case performance of gradient descent on g(Mx) with fixed step-sizes ***
        	PEPit guarantee:	    f(x_n)-f_* <= 0.163818 ||x_0 - x_*||^2
        	Theoretical guarantee:	 f(x_n)-f_* <= 0.16379 ||x_0 - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()
    
    # Declare a strongly convex smooth function G = g(y) and a linear operator M = Mx
    G = problem.declare_function(SmoothStronglyConvexFunction, mu=mug, L=Lg) # g(y)
    if typeM == "gen":
        M = problem.declare_function(function_class=LinearOperator, L=LM)
    elif typeM == "sym":
        M = problem.declare_function(function_class=SymmetricLinearOperator, mu=muM, L=LM)
    elif typeM == "skew":
        M = problem.declare_function(function_class=SkewSymmetricLinearOperator, L=LM)
    
    # Define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Defining unique optimal point xs = x_* of F(x) = g(Mx) and corresponding function value fs = f_*
    xs = Point()                        # xs
    ys = M.gradient(xs)                 # ys = M*xs
    us, fs = G.oracle(ys)               # us = \nabla g(ys) and fs = F(xs) = g(M*ys)
    if typeM == "gen":
        vs = M.gradient_transpose(us)   # vs =  M^T \nabla g(ys) = \nabla F(xs)
    elif typeM == "sym":
        vs = M.gradient(us)             # vs =  M \nabla g(ys) = \nabla F(xs)
    elif typeM == "skew":
        vs = -M.gradient(us)            # vs =  -M \nabla g(ys) = \nabla F(xs)
    problem.add_constraint(vs**2==0)    # vs = \nabla F(xs) = 0

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)
    
    # Run n steps of the GD method
    x = x0
    for _ in range(n):
        y = M.gradient(x)               # y = Mx
        u = G.gradient(y)               # y = \nabla g(y)
        if typeM == "gen":
            v = M.gradient_transpose(u) # v = M^T u
        elif typeM == "sym":
            v = M.gradient(u)           # v = M u
        elif typeM == "skew":
            v = -M.gradient(u)          # v = M^T u = - M u
        x = x - gamma*v

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric(G(M.gradient(x)) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    L = Lg*LM**2
    kappag = mug/Lg
    kappaM = muM/LM
    
    def fun(x):
        return (1-(2*n+1)*x)*(1-x)**(-2*n-1) - 1+kappag

    x = optimize.fsolve(fun, 0.5, xtol=1e-10, full_output=True)[0]
    h0 = x/kappag
    t = np.sqrt(h0[0]/(L*gamma))
    
    if t < kappaM:
        M_star = kappaM
    elif t > 1:
        M_star = 1
    else:
        M_star = t
    
    theoretical_tau = 0.5*L*np.max(( kappag*M_star**2/(kappag-1+(1-kappag*L*gamma*M_star**2)**(-2*n)) ,(1-L*gamma)**(2*n)))

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
    typeM = "sym"
    LM = 1.
    muM = 0.1
    
    pepit_tau, theoretical_tau = wc_gradient_descent_lc(mug=mg, Lg=Lg, typeM=typeM, muM=muM, LM=LM, gamma=1/(Lg*LM**2), n=3, verbose=1)
