import PEPit
from PEPit.functions import SmoothConvexFunction
import math
import warnings


def wc_gradient_descent_silver_stepsize_convex(L, n, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth.

    This code computes a worst-case guarantee for gradient descent of :math:'n' steps following the silver stepsize schedule.
    That is, it computes the smallest possible :math:`\\tau(n, L)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of gradient descent using the silver stepsizes, and
    where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n`, and :math:`L`, :math:`\\tau(n, L)` is computed as the worst-case
    value of :math:`f(x_n)-f_\\star` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma_t \\nabla f(x_t),

    where :math:`\\gamma_t` is a step-size of the :math:'t^{th}' step of the silver step-size schedule.

    **Theoretical guarantee**:
    The theoretical guarantee for the convergence rate of the silver stepsize can be found in [1, Theorem 1.1]:

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{L}{1 + \\sqrt{4\\rho^{2k}-3}} \\|x_0-x_\\star\\|^2.

    **References**:

    `[1] Altschuler, J. M., Parrilo, P. A. (2023). Acceleration by Stepsize Hedging II:
    Silver Stepsize Schedule for Smooth Convex Optimization. arXiv preprint arXiv:2309.16530.
    <https://arxiv.org/abs/2309.16530>`_

    Args:
        L (float): the smoothness constant of the problem.
        n (int): number of iterations (must be a power of 2 minus 1).
        verbose (int): Level of information details to print.
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> L = 10
        >>> pepit_tau, theoretical_tau = pepit_tau, theoretical_tau = wc_gradient_descent_silver_stepsize_convex(L = L, n = 7, verbose=1)
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
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: SCS); optimal value: 0.18421320311999373
        (PEPit) Postprocessing: solver's output is not entirely feasible (smallest eigenvalue of the Gram matrix is: -1.6e-06 < 0).
         Small deviation from 0 may simply be due to numerical error. Big ones should be deeply investigated.
         In any case, from now the provided values of parameters are based on the projection of the Gram matrix onto the cone of symmetric semi-definite matrix.
        (PEPit) Primal feasibility check:
                The solver found a Gram matrix that is positive semi-definite up to an error of 1.6021924382189813e-06
                All the primal scalar constraints are verified up to an error of 5.9084576814194545e-06
        (PEPit) Dual feasibility check:
                The solver found a residual matrix that is positive semi-definite up to an error of 1.7536460910204282e-17
                All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 6.811648931810074e-06
        (PEPit) Final upper bound (dual): 0.18421661438262407 and lower bound (primal example): 0.18421320311999373 
        (PEPit) Duality gap: absolute: 3.4112626303428595e-06 and relative: 1.851801376104846e-05
        *** Example file: worst-case performance of gradient descent with silver step-sizes ***
            PEPit guarantee:	 f(x_n)-f_* <= 0.184217 ||x_0 - x_*||^2
            Theoretical guarantee:	 f(x_n)-f_* <= 0.343775 ||x_0 - x_*||^2

    """
    
    # Setting n if not a power of 2
    if math.ceil(math.log2(n+1)) != math.floor(math.log2(n+1)):
        warnings.warn("Given n not a power of 2 minus 1. Being set as greatest power of two less than given n minus 1.")
        n = 2 ** math.floor(math.log2(n))-1
        
    k = int(math.log2(n+1))
        
    rho = 1 + math.sqrt(2)
    
    

    # Defining silver stepsizes
    h = [1+rho**((k & -k).bit_length()-2) for k in range(1,n+1)]


    # Alternate way of defining silver stepsizes using loop
    #n_glue = int(math.log2(n+1))
    #h = [math.sqrt(2)]
    #    
    #for step in range(n_glue-1):
    #    h = h + [1+rho ** (step)] + h

    # Instantiate PEP
    problem = PEPit.PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothConvexFunction, L=L)

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
        x = x - h[i]/L * func.gradient(x)
        
    # Set the performance metric to the function values accuracy
    problem.set_performance_metric(func(x) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)
    
    theoretical_tau = L/(1+math.sqrt(4*rho**(2*k)-3))
    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent with silver step-sizes ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    # Observe that the theoretical upper bound nearly matches the computational lower bound
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 10
    pepit_tau, theoretical_tau = wc_gradient_descent_silver_stepsize_convex(L = L, n = 7, verbose=1)