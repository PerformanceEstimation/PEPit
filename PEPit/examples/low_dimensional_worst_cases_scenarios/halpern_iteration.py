from PEPit import PEP
from PEPit.operators import LipschitzOperator


def wc_halpern_iteration(n, verbose=1):
    """
    Consider the fixed point problem

    .. math:: \\mathrm{Find}\\, x:\\, x = Ax,

    where :math:`A` is a non-expansive operator,
    that is a :math:`L`-Lipschitz operator with :math:`L=1`.

    This code computes a worst-case guarantee for the **Halpern Iteration**.
    That is, it computes the smallest possible :math:`\\tau(n)` such that the guarantee

    .. math:: \\|x_n - Ax_n\\|^2 \\leqslant \\tau(n) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the **Halpern iteration**,
    and :math:`x_\\star` the fixed point of :math:`A`.

    In short, for a given value of :math:`n`,
    :math:`\\tau(n)` is computed as the worst-case value of
    :math:`\\|x_n - Ax_n\\|^2` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    This example further illustrates how to find a low-dimensional worst-case example using the logdet heuristic [2].
    
    **Algorithm**: The Halpern iteration can be written as

        .. math:: x_{t+1} = \\frac{1}{t + 2} x_0 + \\left(1 - \\frac{1}{t + 2}\\right) Ax_t.

    **Theoretical guarantee**: A **tight** worst-case guarantee for Halpern iteration can be found in [1, Theorem 2.1]:

        .. math:: \\|x_n - Ax_n\\|^2 \\leqslant \\left(\\frac{2}{n+1}\\right)^2 \\|x_0 - x_\\star\\|^2.

    **References**: The detailed approach and tight bound are available in [1].

    `[1] F. Lieder (2021). On the convergence rate of the Halpern-iteration. Optimization Letters, 15(2), 405-418.
    <http://www.optimization-online.org/DB_FILE/2017/11/6336.pdf>`_

    `[2] F. Maryam, H. Hindi, S. Boyd (2003). Log-det heuristic for matrix rank minimization with applications to Hankel
    and Euclidean distance matrices. American Control Conference (ACC).
    <https://web.stanford.edu/~boyd/papers/pdf/rank_min_heur_hankel.pdf>`_

    Args:
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
	>>> pepit_tau, theoretical_tau = wc_halpern_iteration(n=15, verbose=1)
	(PEPit) Setting up the problem: size of the main PSD matrix: 18x18
	(PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
	(PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
	(PEPit) Setting up the problem: interpolation conditions for 1 function(s)
		 function 1 : 272 constraint(s) added
	(PEPit) Setting up the problem: 0 lmi constraint(s) added
	(PEPit) Compiling SDP
	(PEPit) Calling SDP solver
	(PEPit) Solver status: optimal (solver: MOSEK); optimal value: 0.015625108638644202
	(PEPit) Postprocessing: 16 eigenvalue(s) > 0.0017004950517382784 before dimension reduction
	(PEPit) Calling SDP solver
	(PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.015625108638644202
	(PEPit) Postprocessing: 2 eigenvalue(s) > 0.0009050641128584248 after 1 dimension reduction step(s)
	(PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.015625108638644202
	(PEPit) Postprocessing: 2 eigenvalue(s) > 2.24969870935426e-06 after 2 dimension reduction step(s)
	(PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.015625108638644202
	(PEPit) Postprocessing: 1 eigenvalue(s) > 2.51642373983554e-11 after 3 dimension reduction step(s)
	(PEPit) Solver status: optimal (solver: MOSEK); objective value: 0.015625108638644202
	(PEPit) Postprocessing: 1 eigenvalue(s) > 2.51642373983554e-11 after dimension reduction
	*** Example file: worst-case performance of Halpern Iterations ***
		PEPit guarantee:	 ||xN - AxN||^2 <= 0.0156151 ||x0 - x_*||^2
		Theoretical guarantee:	 ||xN - AxN||^2 <= 0.015625 ||x0 - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a non expansive operator
    A = problem.declare_function(LipschitzOperator, L=1.)

    # Start by defining its unique optimal point xs = x_*
    xs, _, _ = A.fixed_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the difference between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of Halpern Iterations
    x = x0
    for i in range(n):
        x = 1 / (i + 2) * x0 + (1 - 1 / (i + 2)) * A.gradient(x)

    # Set the performance metric to distance between xN and AxN
    problem.set_performance_metric((x - A.gradient(x)) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose, dimension_reduction_heuristic="logdet3",tol_dimension_reduction=1e-5)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (2 / (n + 1)) ** 2

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of Halpern Iterations ***')
        print('\tPEPit guarantee:\t ||xN - AxN||^2 <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||xN - AxN||^2 <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_halpern_iteration(n=10, verbose=1)
