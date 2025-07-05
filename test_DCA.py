from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.functions import SmoothFunction

from PEPit.functions import LojasiewiczSmoothFunction
from PEPit.functions import Refined_LojasiewiczSmoothFunction
from PEPit.functions import ExpertRefined_LojasiewiczSmoothFunction


from PEPit.primitive_steps import shifted_optimization_step

from PEPit.functions import ConvexIndicatorFunction
from PEPit.operators import CocoerciveStronglyMonotoneOperator
from PEPit.operators import LipschitzStronglyMonotoneOperator
from PEPit.operators import Refined_CocoerciveStronglyMonotoneOperator
from PEPit.operators import Refined_LipschitzStronglyMonotoneOperator
from PEPit.primitive_steps import proximal_step

import numpy as np

    

def wc_difference_of_convex_algorithm(mu1, mu2, L1, L2, n, alpha = 0, wrapper="cvxpy", solver=None, verbose=1):
    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    f1 = problem.declare_function(SmoothStronglyConvexFunction, L=L1, mu=mu1)
    f2 = problem.declare_function(SmoothStronglyConvexFunction, L=L2, mu=mu2)
    F = f1 - f2

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = F.stationary_point()
    Fs = F(xs)
	
    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()
    
    x = x0
    g1x = f1.gradient(x0)
    g2x = f2.gradient(x0)
    
    problem.set_initial_condition( f1(x) - f2(x) - Fs <= 1 )

    
    for i in range(n):
    	problem.set_performance_metric( (g1x-g2x)**2 )
    	problem.add_constraint( Fs <= f1.value(x) - f2.value(x) - 1/2/(L1-mu2) * (g1x-g2x)**2 )
    	y, _, _ = shifted_optimization_step(g2x, f1)
    	x = ( 1 + alpha ) * y - alpha * x
    	g1x, f1x = f1.oracle(x)
    	g2x, f1x = f2.oracle(x)
    	
    problem.set_performance_metric( (g1x-g2x)**2 )
    problem.add_constraint( Fs <= f1.value(x) - f2.value(x) - 1/2/(L1-mu2) * (g1x-g2x)**2 )
    
    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)
    
    Delta = 1;
    A = 2 * (L1 * L2 - mu1 * L2 * (L1 >= L2) - mu2 * L1 * (L2 > L1));
    B = L1 + L2 + mu1 * (L1 / L2 - 3) * (L1 >= L2) + mu2 * (L2 / L1 - 3) * (L2 > L1);
    C = (L1 * L2 - mu1 * L2 * (L1 >= L2) - mu2 * L1 * (L2 > L1)) / (L1 - mu2);
    theory = A * Delta / (B * n + C);
	
    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theory
    

def wc_difference_of_convex_algorithm_loja(mu1, mu2, L1, L2, eta, n, alpha = 0, wrapper="cvxpy", solver=None, verbose=1):
    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    #F = problem.declare_function(Refined_LojasiewiczSmoothFunction, L=L1, mu=eta, alpha = 2*eta/(2*L1+eta))
    F = problem.declare_function(ExpertRefined_LojasiewiczSmoothFunction, L=L1, mu=eta)
    #F = problem.declare_function(LojasiewiczSmoothFunction, L=L1, mu=eta)
    #f1 = problem.declare_function(SmoothStronglyConvexFunction, L=L1, mu=mu1)
    f2 = problem.declare_function(SmoothFunction, L=L2)
    #F = f1 - f2
    f1 = F + f2

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = F.stationary_point()
    Fs = F(xs)
	
    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()
    
    x = x0
    g1x = f1.gradient(x0)
    g2x = f2.gradient(x0)
    
    problem.set_initial_condition( F(x0) - Fs <= 1 )

    
    for i in range(n):
    	y, _, _ = shifted_optimization_step(g2x, f1)
    	x = ( 1 + alpha ) * y - alpha * x
    	g1x, f1x = f1.oracle(x)
    	g2x, f1x = f2.oracle(x)
    	
    problem.set_performance_metric(F(x) - Fs )
    
    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)
    
    theory = (1-eta/L1) / (1+eta/L2);
	
    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theory
    

def wc_optimistic_gradient(n, gamma, beta, mu, wrapper="cvxpy", solver=None, verbose=1):

    # Instantiate PEP
    problem = PEP()

    # Declare an indicator function and a monotone operator
    ind_C = problem.declare_function(ConvexIndicatorFunction)
    F = problem.declare_function(CocoerciveStronglyMonotoneOperator, mu=mu, beta=beta)

    total_problem = F + ind_C

    # Start by defining its unique optimal point xs = x_*
    xs = total_problem.stationary_point()

    # Then define the starting point x0 of the algorithm and its gradient value g0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute n steps of the Proximal Gradient method starting from x0
    x, _, _ = proximal_step(x0, ind_C, gamma)
    xtilde = x
    V = F.gradient(xtilde)
    for _ in range(n):
        previous_xtilde = xtilde
        xtilde, _, _ = proximal_step(x - gamma * V, ind_C, gamma)
        previous_V = V
        V = F.gradient(xtilde)
        x = xtilde + gamma * (previous_V - V)

    # Set the performance metric to the distance between x(n) and x(n-1)
    problem.set_performance_metric((xtilde - previous_xtilde) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = None

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the Optimistic Gradient Method***')
        print('\tPEPit guarantee:\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau
    

def wc_optimistic_gradient_refined(n, gamma, beta, mu, wrapper="cvxpy", solver=None, verbose=1):

    # Instantiate PEP
    problem = PEP()

    # Declare an indicator function and a monotone operator
    ind_C = problem.declare_function(ConvexIndicatorFunction)
    F = problem.declare_function(Refined_CocoerciveStronglyMonotoneOperator, mu=mu, beta=beta)

    total_problem = F + ind_C

    # Start by defining its unique optimal point xs = x_*
    xs = total_problem.stationary_point()

    # Then define the starting point x0 of the algorithm and its gradient value g0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute n steps of the Proximal Gradient method starting from x0
    x, _, _ = proximal_step(x0, ind_C, gamma)
    xtilde = x
    V = F.gradient(xtilde)
    for _ in range(n):
        previous_xtilde = xtilde
        xtilde, _, _ = proximal_step(x - gamma * V, ind_C, gamma)
        previous_V = V
        V = F.gradient(xtilde)
        x = xtilde + gamma * (previous_V - V)

    # Set the performance metric to the distance between x(n) and x(n-1)
    problem.set_performance_metric((xtilde - previous_xtilde) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = None

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the Optimistic Gradient Method***')
        print('\tPEPit guarantee:\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau
    

def wc_optimistic_gradient2(n, gamma, L, mu, wrapper="cvxpy", solver=None, verbose=1):

    # Instantiate PEP
    problem = PEP()

    # Declare an indicator function and a monotone operator
    ind_C = problem.declare_function(ConvexIndicatorFunction)
    F = problem.declare_function(LipschitzStronglyMonotoneOperator, mu=mu, L=L)

    total_problem = F + ind_C

    # Start by defining its unique optimal point xs = x_*
    xs = total_problem.stationary_point()

    # Then define the starting point x0 of the algorithm and its gradient value g0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute n steps of the Proximal Gradient method starting from x0
    x, _, _ = proximal_step(x0, ind_C, gamma)
    xtilde = x
    V = F.gradient(xtilde)
    for _ in range(n):
        previous_xtilde = xtilde
        xtilde, _, _ = proximal_step(x - gamma * V, ind_C, gamma)
        previous_V = V
        V = F.gradient(xtilde)
        x = xtilde + gamma * (previous_V - V)

    # Set the performance metric to the distance between x(n) and x(n-1)
    problem.set_performance_metric((xtilde - previous_xtilde) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = None

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the Optimistic Gradient Method***')
        print('\tPEPit guarantee:\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau
    
    

def wc_optimistic_gradient_refined2(n, gamma, L, mu, wrapper="cvxpy", solver=None, verbose=1):

    # Instantiate PEP
    problem = PEP()

    # Declare an indicator function and a monotone operator
    ind_C = problem.declare_function(ConvexIndicatorFunction)
    F = problem.declare_function(Refined_LipschitzStronglyMonotoneOperator, mu=mu, L=L)

    total_problem = F + ind_C

    # Start by defining its unique optimal point xs = x_*
    xs = total_problem.stationary_point()

    # Then define the starting point x0 of the algorithm and its gradient value g0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute n steps of the Proximal Gradient method starting from x0
    x, _, _ = proximal_step(x0, ind_C, gamma)
    xtilde = x
    V = F.gradient(xtilde)
    for _ in range(n):
        previous_xtilde = xtilde
        xtilde, _, _ = proximal_step(x - gamma * V, ind_C, gamma)
        previous_V = V
        V = F.gradient(xtilde)
        x = xtilde + gamma * (previous_V - V)

    # Set the performance metric to the distance between x(n) and x(n-1)
    problem.set_performance_metric((xtilde - previous_xtilde) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = None

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the Optimistic Gradient Method***')
        print('\tPEPit guarantee:\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau
    
        
if __name__ == "__main__":
    verbose = 0
    
    L1, L2, mu1, mu2, eta = 2., 5., 0, 0, 1
    n = 5
    pepit_tau, theory = wc_difference_of_convex_algorithm(mu1, mu2, L1, L2, n, alpha = 0, wrapper="mosek", solver=None, verbose=verbose)
    print('*** DCA *** ')
    print('\tPEPit guarantee (std inequalities):\t min_(0<=i<=n) ||nabla f_1(x_(i))-nabla f_2(x_(i))||^2_2 <= {:.6} (F(x0)-Fs)'.format(pepit_tau))
    print('\tPEPit guarantee (std inequalities):\t min_(0<=i<=n) ||nabla f_1(x_(i))-nabla f_2(x_(i))||^2_2 <= {:.6} (F(x0)-Fs)'.format(theory))

    pepit_tau, theory = wc_difference_of_convex_algorithm_loja(mu1, mu2, L1, L2, eta, n, alpha = 0, wrapper="mosek", solver=None, verbose=verbose)
    print('*** DCA *** ')
    print('\tPEPit guarantee (std inequalities):\t (F(x1)-Fs)<= {:.6} (F(x0)-Fs)'.format(pepit_tau))
    print('\tTheory guarantee:\t (F(x1)-Fs)<= {:.6} (F(x0)-Fs)'.format(theory))
    
    n, beta, mu = 3, 1, 0.2
    gamma = 1
    pepit_tau, theory  = wc_optimistic_gradient(n, gamma, beta, mu, wrapper="mosek", solver=None, verbose=verbose)
    pepit_tau_refined, theory  = wc_optimistic_gradient_refined(n, gamma, beta, mu, wrapper="mosek", solver=None, verbose=verbose)
    print('*** OG (Cocoercive) *** ')
    print('\tPEPit guarantee (std inequalities):\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
    print('\tPEPit guarantee (refined inequalities):\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau_refined))
    
    n, L, mu = 3, 1, 0.2
    gamma = 1/np.sqrt(2)
    pepit_tau, theory  = wc_optimistic_gradient2(n, gamma, L, mu, wrapper="mosek", solver=None, verbose=verbose)
    pepit_tau_refined, theory  = wc_optimistic_gradient_refined2(n, gamma, L, mu, wrapper="mosek", solver=None, verbose=verbose)
    print('*** OG (Lipschitz) *** ')
    print('\tPEPit guarantee (std inequalities):\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
    print('\tPEPit guarantee (refined inequalities):\t ||x(n) - x(n-1)||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau_refined))
    
    
    
