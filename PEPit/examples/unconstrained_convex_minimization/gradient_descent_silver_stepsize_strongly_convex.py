import PEPit
from PEPit.functions import SmoothStronglyConvexFunction
import math
import warnings


def wc_gradient_descent_silver_stepsize_strongly_convex(kappa, n, verbose=1):
    """
    Consider the strongly convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`1`-smooth and :math:'\\frac{1}{\\kappa}' strongly-convex.

    This code computes a worst-case guarantee for gradient descent of :math:'n' steps following the silver stepsize schedule.
    That is, it computes the smallest possible :math:`\\tau(n, \\kappa)` such that the guarantee

    .. math:: \\|x_n - x_\\star\\|^2 \\leqslant \\tau(n, \\kappa) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of gradient descent using the silver stepsizes, and
    where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n`, and :math:`\\kappa`, :math:`\\tau(n, \\kappa)` is computed as the worst-case
    value of :math:`\\|x_n - x_\\star\\|^2` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma_t \\nabla f(x_t),

    where :math:`\\gamma_t` is a step-size of the :math:'t^{th}' step of the silver step-size schedule.

    **Theoretical guarantee**:
    The theoretical guarantee for the convergence rate of the silver stepsize can be found in [1, Theorem 4.1]:
    When :math:`n \\leq n^\\star` the garuantee is given by 

    .. math:: \\|x_n - x_\\star\\|^2 \\leqslant e^{-n^\\log_2(\\rho)/\\kappa} \\|x_0-x_\\star\\|^2,

    When :math:`n > n^\\star` the garuantee is given by 
    
    .. math:: \\|x_n - x_\\star\\|^2 \\leqslant e^{-\\frac{n}{n^*} (n^*)^\\log_2(\\rho)/\\kappa} \\|x_0-x_\\star\\|^2

    **References**:

    `[1] Altschuler, J. M., Parrilo, P. A. (2023). Acceleration by Stepsize Hedging I:
    Multi-Step Descent and the Silver Stepsize Schedule. arXiv preprint arXiv:2309.07879.
    <https://arxiv.org/abs/2309.07879>`_

    Args:
        kappa (float): the condition number of the problem.
        n (int): number of iterations (must be a power of 2).
        verbose (int): Level of information details to print.
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> kappa = 100
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_silver_stepsize_strongly_convex(kappa = kappa, n=8, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 10x10
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                    Function 1 : Adding 72 scalar constraint(s) ...
                    Function 1 : 72 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: SCS); optimal value: 0.01698449239366679
        (PEPit) Primal feasibility check:
                The solver found a Gram matrix that is positive semi-definite
                All the primal scalar constraints are verified up to an error of 1.4327414732603905e-06
        (PEPit) Dual feasibility check:
                The solver found a residual matrix that is positive semi-definite up to an error of 1.628693891005565e-15
                All the dual scalar values associated with inequality constraints are nonnegative up to an error of 7.697508620658037e-18
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 0.00010098436068739503
        (PEPit) Final upper bound (dual): 0.016978316564473066 and lower bound (primal example): 0.01698449239366679 
        (PEPit) Duality gap: absolute: -6.1758291937247245e-06 and relative: -0.00036361576493316804
        *** Example file: worst-case performance of gradient descent with silver step-sizes ***
            PEPit guarantee:	 ||x_n - x_*||^2 <= 0.0169783 ||x_0 - x_*||^2
            Theoretical guarantee:	 ||x_n - x_*||^2 <= 0.0169783 ||x_0 - x_*||^2
    """
    
    # Setting n if not a power of 2
    if math.ceil(math.log2(n)) != math.floor(math.log2(n)):
        warnings.warn("Given n not a power of 2. Recursively using sequences of length 2^k for k in the binary expansion of n. Note: the theoretical garauntee will be an overestimate.")
        n = int(n)
        n_glue_list = [i for i in range(n.bit_length()) if n & (1 << i)]

    else:
        n_glue_list = [int(math.log2(n))]   

    
        
    rho = 1 + math.sqrt(2)
    
    theoretical_tau = 1
    h = []
    
    for n_glue in n_glue_list:
        # Defining silver stepsizes
        
        psi = lambda t: (1+kappa*t)/(1+t)
        y = [1/kappa]
        z = [1/kappa]
        
        a = [psi(y[0])]
        b = [psi(z[0])]
        
        h_temp = [b[0]]
        for step in range(n_glue):
            z_old = z[step]
            eta = 1 - z_old
            y_new = z_old/(eta+math.sqrt(1+eta**2))
            z_new = z_old*(eta+math.sqrt(1+eta**2))
            y.append(y_new)
            z.append(z_new)
            a_new = psi(y_new)
            b_new = psi(z_new)
            a.append(a_new)
            b.append(b_new)
            h_tilde = h_temp[:-1]
            h_temp = h_tilde + [a_new]  + h_tilde + [b_new]
        h = h + h_temp
    
        # Get the theoretical rate
        theoretical_tau *= ((1 - z[-1])/(1 + z[-1])) ** 2
        
        
    # Instantiate PEP
    problem = PEPit.PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=1/kappa, L=1)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition(  (x0 - xs) ** 2 <= 1)
   
    # Run n steps of the GD method
    x = x0
    for i in range(n):
        x = x - h[i] * func.gradient(x)
        
    # Set the performance metric to the function values accuracy
    problem.set_performance_metric((x-xs) **2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)
    

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent with silver step-sizes ***')
        print('\tPEPit guarantee:\t ||x_n - x_*||^2 <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||x_n - x_*||^2 <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    # Observe that the theoretical upper bound exactly matches the computational lower bound 
    # (when n is a power of 2, which is the setting that the paper focuses on)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    kappa = 10
    pepit_tau, theoretical_tau = wc_gradient_descent_silver_stepsize_strongly_convex(kappa = kappa, n=8, verbose=1)