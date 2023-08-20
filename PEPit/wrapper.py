import numpy as np

from PEPit.expression import Expression
from PEPit.point import Point

class Wrapper(object):
    """
    A :class:`Wrapper` object interfaces PEPit with an SDP solver (or modelling language).

    Warnings:
        This class must be overwritten by a child class that encodes all particularities of the solver.
        In particular, the methods: send_constraint_to_solver, send_lmi_constraint_to_solver, generate_problem,
        get_dual_variables, get_primal_variables, eval_constraint_dual_values, solve, prepare_heuristic, and heuristic
        must be overwritten.
    

    Attributes:
        _list_of_constraints_sent_to_solver (list): list of :class:`Constraint` and :class:`PSDMatrix` objects associated to the PEP.
                                                    This list does not contain constraints due to internal representation of the 
                                                    problem by the solver.
        _list_of_constraints_sent_to_solver_full (list): full list of constraints associated to the solver.
        prob (): instance of the problem (whose type depends on the solver)
        optimal_G (numpy.array): Gram matrix of the PEP after solving.
        optimal_F (numpy.array): Elements of F after solving.
        optimal_dual (list): Optimal dual variables after solving (same ordering as that of _list_of_constraints_sent_to_solver)
        verbose (bool): verbosity:

                            - 0: No verbose at all
                            - 1: PEPit information is printed but not CVXPY's
                            - 2: Both PEPit and CVXPY details are printed

    """
    def __init__(self):
        """
        :class:`Wrapper` object should not be instantiated as such: only children class should.
        This function initialize all internal variables of the class.

        """
        # Initialize lists of constraints that are used to solve the SDP.
        # Those lists should not be updated by hand, only the solve method does update them.
        self._list_of_constraints_sent_to_solver = list()
        self._list_of_constraints_sent_to_solver_full = list() ##MUST USE FOR EVALUATION TRUE DUAL OBJ!!
        self.optimal_F = None
        self.optimal_G = None
        self.optimal_dual = list()
        self.prob = None
        self.verbose = False

    def check_license(self):
        """
        Check that there is a valid available license for the solver.

        Returns:
            license (bool): is there a valid license?

        """
    
        raise NotImplementedError("This method must be overwritten in children classes")
    
    def _expression_to_matrices(self, expression):
        """
        Translate an expression from an :class:`Expression` to a matrix, a vector, and a constant such that
        
            ..math:: \\mathrm{Tr}(\\text{Gweights}\\,G) + \\text{Fweights}^T F + \\text{cons}
        
        corresponds to the expression.

        Args:
            expression (Expression): any expression.

        Returns:
            Gweights (numpy array): weights of the entries of G in the :class:`Expression`.
            Fweights (numpy array): weights of the entries of F in the :class:`Expression`
            cons (float): constant term in the :class:`Expression`

        """
        cons = 0
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
                    Fweights[key.counter] = weight
                # Inner products are stored in G
                elif type(key) == tuple:
                    point1, point2 = key
                    assert point1.get_is_leaf()
                    assert point2.get_is_leaf()
                    Gweights[point1.counter, point2.counter] = weight
                # Constants are simply constants
                elif key == 1:
                    cons = weight
                # Others don't exist and raise an Exception
                else:
                    raise TypeError("Expressions are made of function values, inner products and constants only!")

        Gweights = (Gweights + Gweights.T)/2
        
        return Gweights, Fweights, cons
        
    def _expression_to_sparse_matrices(self, expression):
        """
        Translate an expression from an :class:`Expression` to a matrix, a vector, and a constant such that
        
            ..math:: \\mathrm{Tr}(\\text{Gweights}\\,G) + \\text{Fweights}^T F + \\text{cons}
            
        where :math:`\\text{Gweights}` and :math:`\\text{Fweights}` are expressed in sparse formats.

        Args:
            expression (Expression): any expression.

        Returns:
            Gweights_indi (numpy array): Set of line indices for the sparse representation of the constraint matrix (multiplying G).
            Gweights_indj (numpy array): Set of column indices for the sparse representation of the constraint matrix (multiplying G).
            Gweights_val (numpy array): Set of values for the sparse representation of the constraint matrix (multiplying G).
            Fweights_ind (numpy array): Set of indices for the sparse representation of the constraint vector (multiplying F).
            Fweights_val (numpy array): Set of values of the sparse representation of the constraint vector (multiplying F).
            cons_val (float): Constant part of the constraint.

        """
        cons_val = 0
        Fweights_ind = list()
        Fweights_val = list()
        Gweights_indi = list()
        Gweights_indj = list()
        Gweights_val = list()

        # If simple function value, then simply return the right coordinate in F
        if expression.get_is_leaf():
            Fweights_ind.append(expression.counter)
            Fweights_val.append(1)
        # If composite, combine all the cvxpy expression found from leaf expressions
        else:
            for key, weight in expression.decomposition_dict.items():
                # Function values are stored in F
                if type(key) == Expression:
                    assert key.get_is_leaf()
                    Fweights_ind.append(key.counter)
                    Fweights_val.append(weight)
                # Inner products are stored in G
                elif type(key) == tuple:
                    point1, point2 = key
                    assert point1.get_is_leaf()
                    assert point2.get_is_leaf()
                    
                    weight_sym = 0 # weight of the symmetrical entry
                    if (point2,point1) in expression.decomposition_dict:
                        if point1.counter >= point2.counter: # if both entry and symmetrical entry: only append in one case
                            weight_sym = expression.decomposition_dict[(point2,point1)]
                            Gweights_val.append((weight+weight_sym)/2)
                            Gweights_indi.append(point1.counter)
                            Gweights_indj.append(point2.counter)
                    else:
                        Gweights_val.append((weight+weight_sym)/2)
                        Gweights_indi.append(max(point1.counter,point2.counter))
                        Gweights_indj.append(min(point1.counter,point2.counter))
                # Constants are simply constants
                elif key == 1:
                    cons_val = weight
                # Others don't exist and raise an Exception
                else:
                    raise TypeError("Expressions are made of function values, inner products and constants only!")


        Fweights_ind = np.array(Fweights_ind)
        Fweights_val = np.array(Fweights_val)
        Gweights_indi = np.array(Gweights_indi)
        Gweights_indj = np.array(Gweights_indj)
        Gweights_val = np.array(Gweights_val)
        
        return Gweights_indi, Gweights_indj, Gweights_val, Fweights_ind, Fweights_val, cons_val

    def _expression_to_sparse_matrices_original(self, expression):
        """
        Translate an expression from an :class:`Expression` to a matrix, a vector, and a constant such that
        
            ..math:: \\mathrm{Tr}(\\text{Gweights}\\,G) + \\text{Fweights}^T F + \\text{cons}
            
        where :math:`\\text{Gweights}` and :math:`\\text{Fweights}` are expressed in sparse formats.

        Args:
            expression (Expression): any expression.

        Returns:
            Gweights_indi (numpy array): Set of line indices for the sparse representation of the constraint matrix (multiplying G).
            Gweights_indj (numpy array): Set of column indices for the sparse representation of the constraint matrix (multiplying G).
            Gweights_val (numpy array): Set of values for the sparse representation of the constraint matrix (multiplying G).
            Fweights_ind (numpy array): Set of indices for the sparse representation of the constraint vector (multiplying F).
            Fweights_val (numpy array): Set of values of the sparse representation of the constraint vector (multiplying F).
            cons_val (float): Constant part of the constraint.

        """
        cons_val = 0
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
                    cons_val += weight
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
        
        return Gweights_indi, Gweights_indj, Gweights_val, Fweights_ind, Fweights_val, cons_val
                
    def send_constraint_to_solver(self, constraint):
        """
        Transfer a PEPit :class:`Constraint` to the solver and add the :class:`Constraint` 
        into the tracking lists.

        Args:
            constraint (Constraint): a :class:`Constraint` object to be sent to the solver.

        Raises:
            ValueError if the attribute `equality_or_inequality` of the :class:`Constraint`
            is neither `equality`, nor `inequality`.

        """
        raise NotImplementedError("This method must be overwritten in children classes")

    def send_lmi_constraint_to_solver(self, psd_counter, psd_matrix, verbose):
        """
        Transfer a PEPit :class:`PSDMatrix` (LMI constraint) to the solver
        and add it the tracking lists.

        Args:
            psd_counter (int): a counter useful for the verbose mode.
            psd_matrix (PSDMatrix): a matrix of expressions that is constrained to be PSD.
            verbose (int): Level of information details to print (Override the CVXPY solver verbose parameter).

                            - 0: No verbose at all
                            - 1: PEPit information is printed but not CVXPY's
                            - 2: Both PEPit and solver details are printed

        """
        raise NotImplementedError("This method must be overwritten in children classes")
        
    def get_dual_variables(self):
        """
        Outputs the list of dual variables.
        
        Returns:
            dual_values (list): numerical values of the dual variables (same ordering as that
                                of _list_of_constraints_sent_to_solver.

        """
        return self.optimal_dual
        
    def get_primal_variables(self):
        """
        Outputs the optimal  of dual variables.
        
        Returns:
            optimal_G (numpy.array): numerical Gram matrix of the PEP after solving.
            optimal_F (numpy.array): numerical elements of F after solving.
            
        """
        return self.optimal_G, self.optimal_F
    
    def eval_constraint_dual_values(self):
        """
        Postprocess the output of the solver and associate each constraint of the list 
        _list_of_constraints_sent_to_solver to their corresponding numerical dual variables.
        
        Returns:
            dual_values (list): ###ça ne devrait pas être necessaire!
            residual (np.array):
            dual_objective (float):

        """
    
        raise NotImplementedError("This method must be overwritten in children classes")
        return dual_values, residual, dual_objective

    def generate_problem(self, objective):
        """
        Instantiate an optimization model using the solver format, whose objective corresponds to a
        PEPit :class:`Expression` object.

        Args:
            objective (Expression): the objective function of the PEP (to be maximized).
        
        Returns:
            prob (): the PEP in the solver's format.

        """
        raise NotImplementedError("This method must be overwritten in children classes")
        return self.prob
    
    def solve(self, **kwargs):
        """
        Solve the PEP with solver options.

        Args:
            kwargs (keywords, optional): solver specific arguments.
        
        Returns:
            status (string): status of the solution / problem.
            name (string): name of the solver.
            value (float): value of the performance metric after solving.
            problem (): solver-specific model of the PEP.
        
        """
        raise NotImplementedError("This method must be overwritten in children classes")
        
    def prepare_heuristic(self, wc_value, tol_dimension_reduction):
        """
        Add the constraint that the objective stay close to its actual value before using 
        dimension-reduction heuristics. That is, we constrain

        .. math:: \\tau \\leqslant \\text{wc value} + \\text{tol dimension reduction}

        Args:
            wc_value (float): the optimal value of the original PEP.
            tol_dimension_reduction (float): tolerance on the objective for finding
                                             low-dimensional examples.
        
        """
        raise NotImplementedError("This method must be overwritten in children classes")
        #add constraint that tau<= wc_value+tol_dimension
        
    def heuristic(self, weight):
        """
        Change the objective of the PEP, specifically for finding low-dimensional examples.
        We specify a matrix :math:`W` (weight), which will allow minimizing :math:`\\mathrm{Tr}(G\\,W)`.

        Args:
            weight (np.array): weights that will be used in the heuristic.
        
        """
        raise NotImplementedError("This method must be overwritten in children classes")
    
    @staticmethod
    def _translate_status(status):
        """
        Translate the solver-specific status to a PEP-standardize string.

        Args:
            status (): solver specific status.
        
        Returns:
            status (string): PEPit-standardized status.
        
        """
        raise NotImplementedError("This method must be overwritten in children classes")
