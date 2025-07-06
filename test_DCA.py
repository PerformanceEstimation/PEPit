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
    
    
    
