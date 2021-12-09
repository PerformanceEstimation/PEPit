import numpy as np

from PEPit.pep import PEP
from PEPit.functions.convex_function import ConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_fppa(A0, gammas, n, verbose=True):
    """
       Consider the minimization problem

       .. math:: f_\star = \\min_x f(x),

       where :math:`f` is  convex and possibly non-smooth.

       This code computes a worst-case guarantee for **fast proximal point method** (FPP). That is, it computes
       the smallest possible :math:`\\tau(n, A_0, \\gamma)` such that the guarantee

       .. math:: f(x_n) - f_\star \\leqslant \\tau(n, A_0, \\gamma) (f(x_0) - f_\star + \\frac{A_0}{2}  \\| x_0 - x_\star \\|^2)

       is valid, where :math:`x_n` is the output of FPP (with learning rates \\gamma_k at step k) and where :math:`x_\star` is a minimizer of :math:`f` and :math:`A_0` is a positive number.

       In short, for given values of :math:`n`,  :math:`A_0` and :math:`\\gamma`, :math:`\\tau(n)` is computed as the worst-case value
       of :math:`f(x_n)-f_\star` when :math:`f(x_0) - f_\star + \\frac{A_0}{2} \\| x_0 - x_\star \\|^2 \\leqslant 1`, for the following algorithm.

       **Algorithm**:
        
           .. math::
               :nowrap:

               \\begin{eqnarray}
                   y_{k+1} &&= (1-\\alpha_{k} ) x_{k} + \\alpha_{k} v_k \\\\
                   x_{k+1} &&= \\arg\\min_x \\left\\{f(x)+\\frac{1}{2\gamma_k}\\|x-y_{k+1}\\|^2 \\right\\}, \\\\
                   v_{k+1} &&= v_k + \\frac{1}{\\alpha_{k}} (x_{k+1}-y_{k+1}) 
               \\end{eqnarray}

       with

           .. math::
               :nowrap:

               \\begin{eqnarray}
                   \\alpha_{k} &&= \\frac{\\sqrt{(A_k \\gamma_k)^2 + 4 A_k \\gamma_k} - A_k \\gamma_k }{2} \\\\
                   A_{k+1} &&= (1 - \\alpha_{k})  A_k
               \\end{eqnarray}
               
       **Theoretical guarantee**:
       A theoretical upper-bound can be found in [1, Theorem 2.3.]:

       .. math:: f(x_n)-f_\\star \\leqslant \\frac{4}{A_0 (\\sum_{i=1}^n \\sqrt{\\gamma_i})^2}(f(x_0) - f_\star + \\frac{A_0}{2}  \\| x_0 - x_\star \\|^2).

       **References**:
       The fast proximal point was analyzed in the following work:
       
       [1] O. Güler. New proximal point algorithms for convex minimization.
        SIAM Journal on Optimization, 2(4):649–664, 1992.
        

       Args:
           A0 (float): initial value for parameter A_0.
           gammas (list): sequence of step sizes. 
           n (int): number of iterations.
           verbose (bool): if True, print conclusion

       Returns:
           tuple: worst_case value, theoretical value


       Example:
           >>> pepit_tau, theoretical_tau = wc_fppa(A0=5, gammas=[(i + 1) / 1.1 for i in range(3)], n=3, verbose=True)
           (PEP-it) Setting up the problem: size of the main PSD matrix: 6x6
           (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
           (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
           (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                     function 1 : 20 constraint(s) added
           (PEP-it) Compiling SDP
           (PEP-it) Calling SDP solver
           (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.01593113594082973
           *** Example file: worst-case performance of optimized gradient method ***
               PEP-it guarantee:       f(x_n)-f_* <= 0.0159311  (f(x_0) - f_\star + A0/2* ||x_0 - x_\star||^2)
               Theoretical guarantee:  f(x_n)-f_* <= 0.0511881  (f(x_0) - f_\star + A0/2* ||x_0 - x_\star||^2)
       """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function
    func = problem.declare_function(ConvexFunction, param={})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is a well-chosen distance between x0 and x^*
    problem.set_initial_condition(func.value(x0) - fs + A0 / 2 * (x0 - xs) ** 2 <= 1)

    # Run the fast proximal point method
    x, v = x0, x0
    A = A0
    for i in range(n):
        alpha = (np.sqrt((A * gammas[i]) ** 2 + 4 * A * gammas[i]) - A * gammas[i]) / 2
        y = (1 - alpha) * x + alpha * v
        x, _, _ = proximal_step(y, func, gammas[i])
        v = v + 1 / alpha * (x - y)
        A = (1 - alpha) * A

    # Set the performance metric to the final distance to optimum in function values
    problem.set_performance_metric(func.value(x) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    accumulation = 0
    for i in range(n):
        accumulation += np.sqrt(gammas[i])
    theoretical_tau = 4 / A0 / accumulation ** 2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of fast proximal point method ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} (f(x_0) - f_\star + A/2* ||x_0 - x_\star||^2)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0) - f_\star + A/2* ||x_0 - x_\star||^2)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 3
    A0 = 5
    gammas = [(i + 1) / 1.1 for i in range(n)]

    wc = wc_fppa(A0=A0,
                 gammas=gammas,
                 n=n)
