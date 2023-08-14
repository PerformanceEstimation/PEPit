import mosek as mosek
import numpy as np
import sys

from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.constraint import Constraint
from PEPit.psd_matrix import PSDMatrix


class Mosek_wrapper(object):
    def __init__(self):
        """

    Attributes:
                            
        """
        # Initialize lists of constraints that are used to solve the SDP.
        # Those lists should not be updated by hand, only the solve method does update them.
        self._list_of_constraints_sent_to_solver = list()
        
        self.task = mosek.Task() #initiate MOSEK's task
        # optimal_F, optimal_G

    def _expression_to_solver(expression):
        """
        Create a sparse matrix representation from an :class:`Expression`.
        TODOTODOTODO: direct faire créer la matrice sparse (sans passer par matrice complète)

        Args:
            expression (Expression): any expression.
            F (cvxpy Variable): a vector representing the function values.
            G (cvxpy Variable): a matrix representing the Gram matrix of all leaf :class:`Point` objects.

        Returns:
            Ai (sp): ...
            Aj (sp): ...
            Aval (sp): ...
            ai (sp): ...
            aval (sp): ...
            alpha_val (float): ...
            

        """
        alpha_val = 0
        Fweights = np.zeros((Expression.counter,))
        Gweights = np.zeros((Point.counter, Point.counter))

        # If simple function value, then simply return the right coordinate in F
        if expression.get_is_leaf():
            Fweights[expression.counter] += 1
        # If composite, combine all the cvxpy expression found from leaf expressions
        else:
            for key, weight in expression.decomposition_dict.items():
                # Function values are stored in F
                if type(key) == Expression:
                    assert key.get_is_leaf()
                    Fweights[key.counter] += weight
                # Inner products are stored in G
                elif type(key) == tuple:
                    point1, point2 = key
                    assert point1.get_is_leaf()
                    assert point2.get_is_leaf()
                    Gweights[point1.counter, point2.counter] += weight
                # Constants are simply constants
                elif key == 1:
                    alpha_val += weight
                # Others don't exist and raise an Exception
                else:
                    raise TypeError("Expressions are made of function values, inner products and constants only!")

        Gweights = (Gweights + Gweights.T)/2
        
        No_zero_ele =np.argwhere(Fweights)
        Fweights_ind = No_zero_ele.flatten()
        Fweights_val = Fweights[Fweights_ind]
        
        No_zero_ele =np.argwhere(np.tril(Gweights))
        Gweights_indi = No_zero_ele[:,0]
        Gweights_indj = No_zero_ele[:,1]
        Gweights_val = Gweights[Gweights_indi,Gweights_indj]
        
        No_zero_ele =np.argwhere(Fweights)
        Fweights_ind = No_zero_ele.flatten()
        Fweights_val = Fweights[Fweights_ind]
        
        # Return the input expression in sparse SDP form
        return Gweights_indi, Gweights_indj, Gweights_val, Fweights_ind, Fweights_val, alpha_val

    def send_constraint_to_solver(self, constraint):
        """
        Add a PEPit :class:`Constraint` in a Mosek task and add it to the tracking list.

        Args:
            constraint (Constraint): a :class:`Constraint` object to be sent to MOSEK.
            task (mosek.Task): a mosek task for handling the problem.

        Raises:
            ValueError if the attribute `equality_or_inequality` of the :class:`Constraint`
            is neither `equality`, nor `inequality`.

        """

        # Sanity check
        assert isinstance(constraint, Constraint)
        inf = 1.0 # for symbolic purpoposes

        # Add constraint to the attribute _list_of_constraints_sent_to_mosek to keep track of
        # all the constraints that have been sent to mosek as well as the order.
        self._list_of_constraints_sent_to_solver.append(constraint)
        
        # how many constraints in the task so far? This will be the constraint number
        nb_cons = self.task.getnumcon()
        
        # Add a mosek constraint via task
        self.task.appendcons(1)
        A_i, A_j, A_val, a_i, a_val, alpha_val = self._expression_to_solver(constraint.expression)
        
        sym_A = self.task.appendsparsesymmat(Point.counter,A_i,A_j,A_val)
        self.task.putbaraij(nb_cons, 0, [sym_A], [1.0])
        self.task.putaijlist(nb_cons+np.zeros(a_i.shape,dtype=np.int8), a_i, a_val)
        
        # Distinguish equality and inequality
        if constraint.equality_or_inequality == 'inequality':
            self.task.putconbound(nb_cons, mosek.boundkey.up, -inf, -alpha_val)
        elif constraint.equality_or_inequality == 'equality':
            self.task.putconbound(nb_cons, mosek.boundkey.fx, -alpha_val, -alpha_val)
        else:
            # Raise an exception otherwise
            raise ValueError('The attribute \'equality_or_inequality\' of a constraint object'
                             ' must either be \'equality\' or \'inequality\'.'
                             'Got {}'.format(constraint.equality_or_inequality))
                             

    def send_lmi_constraint_to_solver(self, psd_counter, psd_matrix, task, verbose):
        """
        Add a PEPit :class:`Constraint` in a Mosek task and add it to the tracking list.


        Args:
            psd_counter (int): a counter useful for the verbose mode.
            psd_matrix (PSDMatrix): a matrix of expressions that is constrained to be PSD.
            verbose (int): Level of information details to print (Override the CVXPY solver verbose parameter).

                            - 0: No verbose at all
                            - 1: PEPit information is printed but not MOSEK's
                            - 2: Both PEPit and MOSEK details are printed

        """

        # Sanity check
        assert isinstance(psd_matrix, PSDMatrix)

        # Add constraint to the attribute _list_of_constraints_sent_to_mosek to keep track of
        # all the constraints that have been sent to mosek as well as the order.
        self._list_of_constraints_sent_to_solver.append(psd_matrix)

        # Create a symmetric matrix in MOSEK
        size = psd_matrix.shape[0]
        self.task.appendbarvars([size])

        # Store one correspondence constraint per entry of the matrix
        for i in range(psd_matrix.shape[0]):
            for j in range(psd_matrix.shape[1]):
                A_i, A_j, A_val, a_i, a_val, alpha_val = self._expression_to_solver(psd_matrix[i, j])
                # how many constraints in the task so far? This will be the constraint number
                nb_cons = self.task.getnumcon()
                # add a constraint in mosek
                self.task.appendcons(1)
                # in MOSEK format: matrices corresponding to the quadratic part of the expression
                sym_A1 = self.task.appendsparsesymmat(Point.counter,A_i, A_j, A_val) #1/2 because we have to symmetrize the matrix!
                # in MOSEK format: this is the matrix which selects one entry of the matrix to be PSD
                sym_A2 = self.task.appendsparsesymmat(size,[max(i,j)],[min(i,j)],[-.5*(i!=j)-1*(i==j)]) #1/2 because we have to symmetrize the matrix!
                # fill the mosek (equality) constraint 
                self.task.putbaraij(nb_cons, 0, [sym_A1], [1.0])
                self.task.putbaraij(nb_cons, psd_matrix.counter+1, [sym_A2], [1.0])
                self.task.putaijlist(nb_cons+np.zeros(a_i.shape,dtype=np.int8), a_i, a_val)
                self.task.putconbound(nb_cons, mosek.boundkey.fx, -alpha_val, -alpha_val)

        # Print a message if verbose mode activated
        if verbose:
            print('\t\t Size of PSD matrix {}: {}x{}'.format(psd_counter + 1, *psd_matrix.shape))

    def generate_problem(self, objective):
        self.task.putclist([tau.counter], [1.0])
        # Input the objective sense (minimize/maximize)
        self.task.putobjsense(mosek.objsense.maximize)
        #if verbose >= 2:
            #self.task.solutionsummary(mosek.streamtype.msg)
        return self.task
        
    def get_dual_variables(self):
        ## Task.gety() for obtaining dual variables (but not of the PSD constraints)
        ## Task.getbarsj(mosek.soltype.itr, i) for dual associated to i
        assert self._list_of_solver_constraints == self.prob.constraints 
        dual_values = [constraint.dual_value for constraint in self.prob.constraints]
        return dual_values
        
    def get_primal_variables(self):
        return self.optimal_G, self.optimal_F
    
    def associate_dual_variables(self):
        #perform the association between constraints and dual variables
        return true
        
    def prepare_heuristic(self, wc_value, tol_dimension_reduction):
        # Add the constraint that the objective stay close to its actual value
        self._list_of_solver_constraints.append(self.objective >= wc_value - tol_dimension_reduction)
        
    def heuristic(self, weight=1):
        if weight.shape == (1,1): #should be improved --> weight=np.identity(Point.counter) by default, but seems to bug here
            obj = cp.trace(self.G)
            self.prob = cp.Problem(objective=cp.Minimize(obj), constraints=self._list_of_solver_constraints)
        else:
            obj = cp.sum(cp.multiply(self.G, weight))
            self.prob = cp.Problem(objective=cp.Minimize(obj), constraints=self._list_of_solver_constraints)
    
    def solve(self, **kwargs):
        # Solve the problem and print summary
        self.task.optimize(**kwargs)
        self.wc_value = task.getprimalobj(mosek.soltype.itr)
        self.optimal_G = self._get_Gram_from_mosek(task.getbarxj(mosek.soltype.itr, 0), Point.counter)
        xx = task.getxx(mosek.soltype.itr)
        tau = xx[-1]
        self.optimal_F = xx
        prosta = task.getprosta(mosek.soltype.itr)
        return prosta, 'MOSEK', tau
