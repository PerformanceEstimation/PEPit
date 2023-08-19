import mosek as mosek
import numpy as np
import sys

from PEPit.wrapper import Wrapper
from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.constraint import Constraint
from PEPit.psd_matrix import PSDMatrix


class Mosek_wrapper(Wrapper):
    """
    A :class:`Mosek_wrapper` object interfaces PEPit with the SDP solver `MOSEK<https://www.mosek.com/>`_.

    This class overwrittes the :class:`Wrapper` for MOSEK. In particular, it implements the methods:
    send_constraint_to_solver, send_lmi_constraint_to_solver, generate_problem, get_dual_variables,
    get_primal_variables, eval_constraint_dual_values, prepare_heuristic, and heuristic
    
    Attributes:
        _list_of_constraints_sent_to_solver (list): list of :class:`Constraint` and :class:`PSDMatrix` objects associated to the PEP.
                                                    This list does not contain constraints due to internal representation of the 
                                                    problem by the solver.
        _list_of_constraints_sent_to_solver_full (list): full list of constraints associated to the solver.
        prob (mosek.task): instance of the problem.
        optimal_G (numpy.array): Gram matrix of the PEP after solving.
        optimal_F (numpy.array): Elements of F after solving.
        optimal_dual (list): Optimal dual variables after solving (same ordering as that of _list_of_constraints_sent_to_solver)
        verbose (bool): verbosity:

                            - 0: No verbose at all
                            - 1: PEPit information is printed but not MOSEK's
                            - 2: Both PEPit and CVXPY details are printed

    """
    def __init__(self, verbose=False):
        """
        This function initialize all internal variables of the class. 
        
        Args:
            verbose (bool): verbose mode of the solver.

        """
        # Initialize lists of constraints that are used to solve the SDP.
        # Those lists should not be updated by hand, only the solve method does update them.
        self._list_of_constraints_sent_to_solver = list()
        self._constraint_index_in_mosek = list() # index of the MOSEK constraint for each element of the previous list
        self._nb_pep_constraints_in_mosek = 0
        self._list_of_psd_constraints_sent_to_solver = list()
        self._nb_pep_SDPconstraints_in_mosek = 1
        self._list_of_constraints_sent_to_solver_full = list()
        self.verbose = verbose
        
        self.env = mosek.Env()
        self.task = self.env.Task() #initiate MOSEK's task
        
        if verbose >1 :
            self.task.set_Stream(mosek.streamtype.log, self.streamprinter)
        
        # "Initialize" the Gram matrix
        self.task.appendbarvars([Point.counter])
        # "Initialize" function value variables (an additional variable "tau" is for handling the objective, hence +1)
        self.task.appendvars(Expression.counter+1) 
            
        inf = 1.0 # symbolical purposes
        for i in range(Expression.counter+1):
            self.task.putvarbound(i, mosek.boundkey.fr, -inf, +inf) # no bounds on function values (nor on tau)


    def check_license(self):
        """
        Check that there is a valid available license for MOSEK.

        Returns:
            license (bool): is there a valid license?
            
        """
        try:
            self.env.checkoutlicense(mosek.feature.pton)
        except:
            return False
            
        return self.env.expirylicenses() >= 0 # number of days until license expires >= 0?
        
    @staticmethod
    def _expression_to_solver(expression):
        """
        Create a sparse matrix representation from an :class:`Expression`.

        Args:
            expression (Expression): any expression.

        Returns:
            Ai (numpy array): Set of line indices for the sparse representation of the constraint matrix (multiplying G).
            Aj (numpy array): Set of column indices for the sparse representation of the constraint matrix (multiplying G).
            Aval (numpy array): Set of values for the sparse representation of the constraint matrix (multiplying G).
            ai (numpy array): Set of indices for the sparse representation of the constraint vector (multiplying F).
            aval (numpy array): Set of values of the sparse representation of the constraint vector (multiplying F).
            alpha_val (float): Constant part of the constraint.

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

    def send_constraint_to_solver(self, constraint, track=True):
        """
        Add a PEPit :class:`Constraint` in a Mosek task and add it to the tracking list.

        Args:
            constraint (Constraint): a :class:`Constraint` object to be sent to MOSEK.
            track (bool, optional): do we track the constraint (saving dual variable, etc.)?

        Raises:
            ValueError if the attribute `equality_or_inequality` of the :class:`Constraint`
            is neither `equality`, nor `inequality`.

        """

        # Sanity check
        assert isinstance(constraint, Constraint)
        inf = 1.0 # for symbolic purpoposes

        # Add constraint to the attribute _list_of_constraints_sent_to_mosek to keep track of
        # all the constraints that have been sent to mosek as well as the order.
        if track:
            self._list_of_constraints_sent_to_solver.append(constraint)
            self._nb_pep_constraints_in_mosek += 1
        
        # how many constraints in the task so far? This will be the constraint number
        nb_cons = self.task.getnumcon()
        
        # Add a mosek constraint via task
        self.task.appendcons(1)
        A_i, A_j, A_val, a_i, a_val, alpha_val = self._expression_to_solver(constraint.expression)
        
        sym_A = self.task.appendsparsesymmat(Point.counter,A_i,A_j,A_val)
        self.task.putbaraij(nb_cons, 0, [sym_A], [1.0])
        self.task.putaijlist(nb_cons+np.zeros(a_i.shape,dtype=np.int8), a_i, a_val)
        
        if track:
            self._constraint_index_in_mosek.append(nb_cons)
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
                             

    def send_lmi_constraint_to_solver(self, psd_counter, psd_matrix):
        """
        Transfer a PEPit :class:`PSDMatrix` (LMI constraint) to MOSEK and add it the tracking lists.

        Args:
            psd_counter (int): a counter useful for the verbose mode.
            psd_matrix (PSDMatrix): a matrix of expressions that is constrained to be PSD.

        """

        # Sanity check
        assert isinstance(psd_matrix, PSDMatrix)

        # Add constraint to the attribute _list_of_constraints_sent_to_mosek to keep track of
        # all the constraints that have been sent to mosek as well as the order.
        self._list_of_constraints_sent_to_solver.append(psd_matrix)
        self._nb_pep_SDPconstraints_in_mosek += 1

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
        if self.verbose > 0:
            print('\t\t Size of PSD matrix {}: {}x{}'.format(psd_counter + 1, *psd_matrix.shape))
            
##commented + specified until here    
    def eval_constraint_dual_values(self):
        """
        Postprocess the output of the solver and associate each constraint of the list 
        _list_of_constraints_sent_to_solver to their corresponding numerical dual variables.
        
        Returns:
            dual_values (list): ###ça ne devrait pas être necessaire!
            residual (np.array):
            dual_objective (float):

        """
        ## Task.gety() for obtaining dual variables (but not of the PSD constraints)
        ## Task.getbarsj(mosek.soltype.itr, i) for dual associated to i
        
        #MUST CHECK the nb of constraints, somehow
        scalar_dual_values = self.task.gety(mosek.soltype.itr)
        dual_values = list()
        dual_objective = 0.
        counter_psd = 1
        counter_scalar = 0
        dual_values.append(self._get_Gram_from_mosek(self.task.getbarsj(mosek.soltype.itr, 0), Point.counter))
        for constraint_or_psd in self._list_of_constraints_sent_to_solver:
            if isinstance(constraint_or_psd, Constraint):
                #note: p-e dangereux? je fais l'hypothèse qu'on parcourir la liste des cons dans le bon ordre
                dual_values.append(scalar_dual_values[self._constraint_index_in_mosek[counter_scalar]])
                constraint_or_psd._dual_variable_value = dual_values[-1]
                counter_scalar += 1
                constraint_dict = constraint_or_psd.expression.decomposition_dict
                if (1 in constraint_dict):
                    dual_objective -= dual_values[-1] * constraint_dict[1] ## ATTENTION: on ne tient pas compte des constantes dans les LMIs en faisant juste ça!!
            elif isinstance(constraint_or_psd, PSDMatrix):
                dual_values.append(self._get_Gram_from_mosek(self.task.getbarsj(mosek.soltype.itr, counter_psd), constraint_or_psd.shape[0]))
                assert dual_values[-1].shape == constraint_or_psd.shape
                constraint_or_psd._dual_variable_value = dual_values[-1]
                counter_psd += 1
                
        assert len(dual_values)-1 == len(self._list_of_constraints_sent_to_solver) #-1 because we added the dual corresponding to the Gram matrix
        residual = dual_values[0]

        return dual_values, residual, dual_objective
        
    def prepare_heuristic(self, wc_value, tol_dimension_reduction):
        # Add the constraint that the objective stay close to its actual value
        self.task.putclist([Expression.counter-1], [0.0])
        self.task.putobjsense(mosek.objsense.minimize)
        self.send_constraint_to_solver(self.objective >= wc_value - tol_dimension_reduction, track = False)
        
    def heuristic(self, weight=np.identity(Point.counter)):
        No_zero_ele =np.argwhere(np.tril(weight))
        W_i = No_zero_ele[:,0]
        W_j = No_zero_ele[:,1]
        W_val = weight[W_i, W_j]
        sym_W = self.task.appendsparsesymmat(Point.counter,W_i,W_j,W_val)
        self.task.putbarcj(0,[sym_W],[1.0]) 
        self.task.putobjsense(mosek.objsense.minimize)

    def generate_problem(self, objective):
        #task.putclist([tau.counter], [0.0])
        assert self.task.getmaxnumvar() == Expression.counter
        self.objective = objective
        _, _, _, Fweights_ind, Fweights_val, _ = self._expression_to_solver(objective)
        self.task.putclist(Fweights_ind, Fweights_val) #to be cleaned by calling _expression_to_solver(objective)?
        # Input the objective sense (minimize/maximize)
        self.task.putobjsense(mosek.objsense.maximize)
        if self.verbose > 1:
            self.task.solutionsummary(mosek.streamtype.msg)
        return self.task
    
    def solve(self, **kwargs):
        # Solve the problem and print summary
        self.task.optimize(**kwargs)
        self.wc_value = self.task.getprimalobj(mosek.soltype.itr)
        self.optimal_G = self._get_Gram_from_mosek(self.task.getbarxj(mosek.soltype.itr, 0), Point.counter)
        xx = self.task.getxx(mosek.soltype.itr)
        tau = xx[-1]
        self.optimal_F = xx
        prosta = self.task.getprosta(mosek.soltype.itr)
        return prosta, 'MOSEK', tau, self.task
        
    @staticmethod
    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()
    
    @staticmethod
    def _get_Gram_from_mosek(tril, size):
        # MOSEK returns:
        # the primal solution for a semidefinite variable. Only the lower triangular part of
        # is returned because the matrix by construction is symmetric. The format is that the columns are stored sequentially in the natural order.
        G = np.zeros((size,size))
        counter = 0
        for j in range(size):
            for i in range(size-j):
                G[j+i,j] = tril[counter]
                G[j,j+i] = tril[counter]
                counter += 1
        return G
