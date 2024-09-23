from numpy import array
from PEPit import PEP
from PEPit.operators import LipschitzStronglyMonotoneOperator, StronglyMonotoneOperator
from PEPit.examples.monotone_inclusions_variational_inequalities import wc_frugal_resolvent_splitting
from PEPit.examples.monotone_inclusions_variational_inequalities import wc_reduced_frugal_resolvent_splitting


def wc_reduced_frugal_resolvent_splitting_dr(l, mu, alpha, gamma, wrapper="cvxpy", solver=None, verbose=1):
    """
    See description in `PEPit/examples/monotone_inclusions_variational_inequalities/reduced_frugal_resolvent_splitting.py`.
    This example is for testing purposes. It validates that the reduced frugal resolvent splitting with the matrices below correctly gives the worst-case value for Douglas-Rachford splitting with :math:`l`-Lipschitz and maximally monotone and (maximally) :math:`\\mu`-strongly
    monotone operators.

    Args:
        l (float): the Lipschitz parameter.
        mu (float): the strongly monotone parameter.    
        alpha (float): resolvent scaling parameter.
        gamma (float): step size parameter.
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): level of information details to print.

                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + solver details.

    Returns:
        pepit_tau (float): worst-case value
    """
    problem = PEP()
    problem.declare_function(StronglyMonotoneOperator, mu=mu)
    problem.declare_function(LipschitzStronglyMonotoneOperator, L=l, mu=0)
    wc = wc_reduced_frugal_resolvent_splitting(
                L=array([[0,0],[2,0]]), 
                M=array([[1,-1]]),
                problem=problem,
                alpha=alpha,
                gamma=gamma,
                wrapper=wrapper,
                solver=solver,
                verbose=verbose)
    return wc

def wc_frugal_resolvent_splitting_dr(l, mu, alpha, gamma, wrapper="cvxpy", solver=None, verbose=1):
    """
    See description in `PEPit/examples/monotone_inclusions_variational_inequalities/frugal_resolvent_splitting.py`.
    This example is for testing purposes. It validates that the frugal resolvent splitting with the matrices below correctly gives the worst-case value for Douglas-Rachford splitting with :math:`l`-Lipschitz and maximally monotone and (maximally) :math:`\\mu`-strongly
    monotone operators.

    Args:
        l (float): the Lipschitz parameter.
        mu (float): the strongly monotone parameter.    
        alpha (float): resolvent scaling parameter.
        gamma (float): step size parameter.
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): level of information details to print.

                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + solver details.

    Returns:
        pepit_tau (float): worst-case value
    """
    problem = PEP()
    problem.declare_function(StronglyMonotoneOperator, mu=mu)
    problem.declare_function(LipschitzStronglyMonotoneOperator, L=l, mu=0)
    wc = wc_frugal_resolvent_splitting(
                L=array([[0,0],[2,0]]), 
                W=array([[1,-1],[-1,1]]),
                problem=problem,
                alpha=alpha,
                gamma=gamma,
                wrapper=wrapper,
                solver=solver,
                verbose=verbose)
    return wc
