import cvxpy as cp
import numpy as np

from PEPit.wrapper import Wrapper
from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.constraint import Constraint
from PEPit.psd_matrix import PSDMatrix


class Cvxpy_wrapper(Wrapper):
    """
    A :class:`Cvxpy_wrapper` object interfaces PEPit with the `CVXPY<https://www.cvxpy.org/>`_ modelling language.

    This class overwrittes the :class:`Wrapper` for CVXPY. In particular, it implements the methods:
    send_constraint_to_solver, send_lmi_constraint_to_solver, generate_problem, get_dual_variables,
    get_primal_variables, eval_constraint_dual_values, solve, prepare_heuristic, and heuristic.
    
    Attributes:
        _list_of_constraints_sent_to_solver (list): list of :class:`Constraint` and :class:`PSDMatrix` objects associated to the PEP.
                                                    This list does not contain constraints due to internal representation of the 
                                                    problem by the solver.
        _list_of_constraints_sent_to_solver_full (list): full list of constraints associated to the solver.
        prob (cvxpy.Problem): instance of the problem.
        optimal_G (numpy.array): Gram matrix of the PEP after solving.
        optimal_F (numpy.array): Elements of F after solving.
        optimal_dual (list): Optimal dual variables after solving (same ordering as that of _list_of_constraints_sent_to_solver)
        F (cvxpy.Variable): CVXPY variable corresponding to leaf :class:`Expression` objects of the PEP.
        G (cvxpy.Variable): CVXPY variable corresponding the Gram matrix of the PEP.
        verbose (bool): verbosity:

                            - 0: No verbose at all
                            - 1: PEPit information is printed but not CVXPY's
                            - 2: Both PEPit and CVXPY details are printed (overwrittes CVXPY's setting)

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
        self._list_of_solver_constraints = list()
        self._list_of_constraints_sent_to_solver_full = list()
        self.F = cp.Variable((Expression.counter+1,)) # need the +1 because the objective will be created afterwards
        self.G = cp.Variable((Point.counter, Point.counter), symmetric=True)

        # Express the constraints from F, G and objective
        # Start with the main LMI condition
        self._list_of_solver_constraints = [self.G >> 0]
        self.verbose = verbose
        
    def check_license(self):
        """
        Check that there is a valid available license for CVXPY.

        Returns:
            license (bool): no license needed: True
            
        """
        return True
        
    def _expression_to_solver(self, expression):
        """
        Create a cvxpy compatible expression from an :class:`Expression`.

        Args:
            expression (Expression): any expression.

        Returns:
            cvxpy_variable (cvxpy Variable): The expression in terms of F and G.

        """
        Gweights, Fweights, cons = self._expression_to_matrices(expression)
        cvxpy_variable = cons + self.F @ Fweights + cp.sum(cp.multiply(self.G, Gweights))

        # Return the input expression in a cvxpy variable
        return cvxpy_variable
    
    def send_constraint_to_solver(self, constraint):
        """
        Transform a PEPit :class:`Constraint` into a CVXPY one
        and add the 2 formats of the constraints into the tracking lists.

        Args:
            constraint (Constraint): a :class:`Constraint` object to be sent to CVXPY.

        Raises:
            ValueError if the attribute `equality_or_inequality` of the :class:`Constraint`
            is neither `equality`, nor `inequality`.

        """

        # Sanity check
        assert isinstance(constraint, Constraint)

        # Add constraint to the attribute _list_of_constraints_sent_to_solver to keep track of
        # all the constraints that have been sent to CVXPY as well as the order.
        self._list_of_constraints_sent_to_solver.append(constraint)

        # Distinguish equality and inequality
        if constraint.equality_or_inequality == 'inequality':
            cvxpy_constraint = self._expression_to_solver(constraint.expression) <= 0
        elif constraint.equality_or_inequality == 'equality':
            cvxpy_constraint = self._expression_to_solver(constraint.expression) == 0
        else:
            # Raise an exception otherwise
            raise ValueError('The attribute \'equality_or_inequality\' of a constraint object'
                             ' must either be \'equality\' or \'inequality\'.'
                             'Got {}'.format(constraint.equality_or_inequality))

        # Add the corresponding CVXPY constraint to the list of constraints to be sent to CVXPY
        self._list_of_solver_constraints.append(cvxpy_constraint)

    def send_lmi_constraint_to_solver(self, psd_counter, psd_matrix):
        """
        Transform a PEPit :class:`PSDMatrix` into a CVXPY symmetric PSD matrix
        and add the 2 formats of the constraints into the tracking lists.

        Args:
            psd_counter (int): a counter useful for the verbose mode.
            psd_matrix (PSDMatrix): a matrix of expressions that is constrained to be PSD.

        """

        # Sanity check
        assert isinstance(psd_matrix, PSDMatrix)

        # Add psd_matrix to the attribute _list_of_constraints_sent_to_solver to keep track of
        # all the constraints that have been sent to CVXPY as well as the order.
        self._list_of_constraints_sent_to_solver.append(psd_matrix)

        # Create a symmetric matrix in CVXPY
        M = cp.Variable(psd_matrix.shape, symmetric=True)

        # Store the lmi constraint
        cvxpy_constraints_list = [M >> 0]

        # Store one correspondence constraint per entry of the matrix
        for i in range(psd_matrix.shape[0]):
            for j in range(psd_matrix.shape[1]):
                cvxpy_constraints_list.append(M[i, j] == self._expression_to_solver(psd_matrix[i, j]))

        # Print a message if verbose mode activated
        if self.verbose > 0:
            print('\t\t Size of PSD matrix {}: {}x{}'.format(psd_counter + 1, *psd_matrix.shape))

        # Add the corresponding CVXPY constraints to the list of constraints to be sent to CVXPY
        self._list_of_solver_constraints += cvxpy_constraints_list

    def eval_constraint_dual_values(self):
        """
        Recover all dual variables and store them in associated :class:`Constraint` and :class:`PSDMatrix` objects.

        Returns:
             dual_values (list): list of dual variables (floats) associated to _list_of_constraints_sent_to_solver (same ordering).
             residual (np.array): main dual PSD matrix (dual to the PSD constraint on the Gram matrix).
             dual_objective (float): numerical value of the dual objective function.

        Raises:
            TypeError if the attribute `_list_of_constraints_sent_to_solver` of this object
            is neither a :class:`Constraint` object, nor a :class:`PSDMatrix` one.

        """
        
        assert self._list_of_solver_constraints == self.prob.constraints
        dual_values = [constraint.dual_value for constraint in self.prob.constraints]
        self.dual_values = dual_values
        # Store residual, dual value of the main lmi
        residual = dual_values[0]
        assert residual.shape == (Point.counter, Point.counter)
        
        # initiate the value of the dual objective (updated below)
        dual_objective = 0.

        # Set counter
        #counter = len(self.list_of_performance_metrics)+1
        counter = 1 # the list of constraints sent to cvxpy contains the perf metrics

        for constraint_or_psd in self._list_of_constraints_sent_to_solver:
            if isinstance(constraint_or_psd, Constraint):
                constraint_or_psd._dual_variable_value = dual_values[counter]
                constraint_dict = constraint_or_psd.expression.decomposition_dict
                if (1 in constraint_dict):
                    dual_objective -= dual_values[counter] * constraint_dict[1] ## ATTENTION: on ne tient pas compte des constantes dans les LMIs en faisant juste ça!!
                counter += 1
            elif isinstance(constraint_or_psd, PSDMatrix):
                assert dual_values[counter].shape == constraint_or_psd.shape
                constraint_or_psd._dual_variable_value = dual_values[counter]
                counter += 1
                size = constraint_or_psd.shape[0] * constraint_or_psd.shape[1]
                constraint_or_psd.entries_dual_variable_value = np.array(dual_values[counter:counter + size]
                                                                         ).reshape(constraint_or_psd.shape)
                counter += size
            else:
                raise TypeError("The list of constraints that are sent to CVXPY should contain only"
                                "\'Constraint\' objects of \'PSDMatrix\' objects."
                                "Got {}".format(type(constraint_or_psd)))

        # Verify nothing is left
        assert len(dual_values) == counter

        # Return the position of the reached performance metric
        return dual_values, residual, dual_objective
        
    def generate_problem(self, objective):
        """
        Instantiate an optimization model using the cvxpy format, whose objective corresponds to a
        PEPit :class:`Expression` object.

        Args:
            objective (Expression): the objective function of the PEP (to be maximized).
        
        Returns:
            prob (cvxpy.Problem): the PEP in cvxpy format.

        """
        self.objective = self._expression_to_solver(objective)
        self.prob = cp.Problem(objective=cp.Maximize(self.objective), constraints=self._list_of_solver_constraints)
        return self.prob
        
    def solve(self, **kwargs):
        """
        Solve the PEP.

        Args:
            kwargs (keywords, optional): solver specific arguments.
        
        Returns:
            status (string): status of the solution / problem.
            name (string): name of the solver.
            value (float): value of the performance metric after solving.
            problem (cvxpy Problem): solver-specific model of the PEP.
        
        """
        if self.verbose > 1:
            kwargs['verbose'] = True
        self.prob.solve(**kwargs)
        self.optimal_G = self.G.value
        self.optimal_F = self.F.value
        return self.prob.status, self.prob.solver_stats.solver_name, self.objective.value, self.prob
        
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
        # Add the constraint that the objective stay close to its actual value
        self._list_of_solver_constraints.append(self.objective >= wc_value - tol_dimension_reduction)
        
    def heuristic(self, weight):
        """
        Change the objective of the PEP, specifically for finding low-dimensional examples.
        We specify a matrix :math:`W` (weight), which will allow minimizing :math:`\\mathrm{Tr}(G\\,W)`.

        Args:
            weight (np.array): weights that will be used in the heuristic.
        
        """
        obj = cp.sum(cp.multiply(self.G, weight))
        self.prob = cp.Problem(objective=cp.Minimize(obj), constraints=self._list_of_solver_constraints)
        return self.prob