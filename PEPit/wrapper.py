import numpy as np

from PEPit.expression import Expression
from PEPit.point import Point
from PEPit.constraint import Constraint
from PEPit.psd_matrix import PSDMatrix


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

    def __init__(self, verbose):
        """
        :class:`Wrapper` object should not be instantiated as such: only children class should.
        This function initialize all internal variables of the class.

        """
        # Initialize lists of constraints that are used to solve the SDP.
        # Those lists should not be updated by hand, only the solve method does update them.
        self._list_of_constraints_sent_to_solver = list()
        self._list_of_constraints_sent_to_solver_full = list()  # MUST USE FOR EVALUATION TRUE DUAL OBJ!!  ## TODO check usage

        self.optimal_F = None
        self.optimal_G = None
        self.objective = None  # PEPit leaf expression
        self.optimal_dual = list()
        self.dual_residual = None
        self.dual_objective = None

        self.prob = None
        self.verbose = verbose

        # feasibility: (i) primal (linear constraints + eigs of the LMI) (ii) dual
        self.primal_feas = None
        self.primal_feas_eigs = None
        self.dual_feas = None
        self.dual_feas_eigs = None

    def setup_environment(self):
        raise NotImplementedError

    def check_license(self):
        """
        Check that there is a valid available license for the solver.

        Returns:
            license (bool): is there a valid license?

        """

        raise NotImplementedError("This method must be overwritten in children classes")

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
            optimal_dual (list): numerical values of the dual variables (same ordering as that
                                 of _list_of_constraints_sent_to_solver).
            dual_residual (np.array): dual variable corresponding to the main (primal) Gram matrix.
            dual_objective (float): dual objective value.

        """
        return self.optimal_dual, self.dual_residual, self.dual_objective

    def get_primal_variables(self):
        """
        Outputs the optimal value of primal variables.
        
        Returns:
            optimal_G (numpy.array): numerical Gram matrix of the PEP after solving.
            optimal_F (numpy.array): numerical elements of F after solving.
            
        """
        return self.optimal_G, self.optimal_F

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
        dual_values, residual = self._recover_dual_values()

        assert residual.shape == (Point.counter, Point.counter)

        # initiate the value of the dual objective (updated below)
        dual_objective = 0.

        # Set counter
        counter = 1

        for constraint_or_psd in self._list_of_constraints_sent_to_solver:
            if isinstance(constraint_or_psd, Constraint):
                constraint_or_psd._dual_variable_value = dual_values[counter]
                constraint_dict = constraint_or_psd.expression.decomposition_dict
                if 1 in constraint_dict:
                    dual_objective -= dual_values[counter] * constraint_dict[1]
                counter += 1
            elif isinstance(constraint_or_psd, PSDMatrix):
                assert dual_values[counter].shape == constraint_or_psd.shape
                constraint_or_psd._dual_variable_value = dual_values[counter]
                counter += 1
                n, m = constraint_or_psd._dual_variable_value.shape
                # update dual objective
                for i in range(n):
                    for j in range(m):
                        constraint_dict = constraint_or_psd.__getitem__((i, j)).decomposition_dict
                        if 1 in constraint_dict:
                            dual_objective += constraint_or_psd._dual_variable_value[i, j] * constraint_dict[1]
            else:
                raise TypeError("The list of constraints that are sent to CVXPY should contain only"
                                "\'Constraint\' objects of \'PSDMatrix\' objects."
                                "Got {}".format(type(constraint_or_psd)))

        self.optimal_dual = dual_values
        self.dual_residual = residual
        self.dual_objective = dual_objective

        # Verify nothing is left
        assert len(dual_values) == counter

        return dual_values, residual, dual_objective

    def _recover_dual_values(self):
        """
        Postprocess the output of the solver and associate each constraint of the list 
        _list_of_constraints_sent_to_solver to their corresponding numerical dual variables.
        
        Returns:
            dual_values (list): 
            residual (np.array):

        """

        raise NotImplementedError("This method must be overwritten in children classes")
        # return dual_values, residual

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
        # add constraint that tau<= wc_value+tol_dimension

    def heuristic(self, weight):
        """
        Change the objective of the PEP, specifically for finding low-dimensional examples.
        We specify a matrix :math:`W` (weight), which will allow minimizing :math:`\\mathrm{Tr}(G\\,W)`.

        Args:
            weight (np.array): weights that will be used in the heuristic.
        
        """
        raise NotImplementedError("This method must be overwritten in children classes")
