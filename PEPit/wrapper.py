from PEPit.expression import Expression
from PEPit.point import Point
from PEPit.constraint import Constraint
from PEPit.psd_matrix import PSDMatrix


class Wrapper(object):
    """
    A :class:`Wrapper` object interfaces PEPit with an SDP solver (or modelling language).

    Warnings:
        This class must be overwritten by a child class that encodes all particularities of the solver.
        In particular, the methods send_constraint_to_solver, send_lmi_constraint_to_solver, generate_problem,
        get_dual_variables, get_primal_variables, eval_constraint_dual_values, solve, prepare_heuristic, and heuristic
        must be overwritten.

    Attributes:
        _list_of_constraints_sent_to_solver (list): list of :class:`Constraint` and :class:`PSDMatrix` objects
                                                    associated to the PEP. This list does not contain constraints
                                                    due to internal representation of the problem by the solver.
        optimal_F (numpy.array): Elements of F after solving.
        optimal_G (numpy.array): Gram matrix of the PEP after solving.
        objective (Expression): The objective expression that must be maximized.
                                This is an additional :class:`Expression` created by the PEP to deal with cases
                                where the user wants to maximize a minimum of several expressions.
        dual_values (list): Optimal dual variables after solving
                            (same ordering as that of _list_of_constraints_sent_to_solver).
        residual (Iterable of Iterables of floats): The residual of the problem, i.e. the dual variable of the Gram.
        prob: instance of the problem (whose type depends on the solver).
        solver_name (str): The name of the solver the wrapper interact with.
        verbose (int): Level of information details to print (Override the solver verbose parameter).

                       - 0: No verbose at all
                       - 1: PEPit information is printed but not solver's
                       - 2: Both PEPit and solver details are printed

    """

    def __init__(self, verbose=1):
        """
        :class:`Wrapper` object should not be instantiated as such: only children class should.
        This function initializes all internal variables of the class.

        Args:
            verbose (int): Level of information details to print (Override the solver verbose parameter).

                           - 0: No verbose at all
                           - 1: PEPit information is printed but not solver's
                           - 2: Both PEPit and solver details are printed

        """
        # Initialize lists of constraints that are used to solve the SDP.
        # Those lists should not be updated by hand, only the solve method does update them.
        self._list_of_constraints_sent_to_solver = list()

        self.optimal_F = None
        self.optimal_G = None
        self.objective = None  # PEPit leaf expression
        self.dual_values = list()
        self.residual = None

        self.prob = None
        self.solver_name = None
        self.verbose = verbose

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

    def send_lmi_constraint_to_solver(self, psd_counter, psd_matrix):
        """
        Transfer a PEPit :class:`PSDMatrix` (LMI constraint) to the solver
        and add it the tracking lists.

        Args:
            psd_counter (int): a counter useful for the verbose mode.
            psd_matrix (PSDMatrix): a matrix of expressions that is constrained to be PSD.

        """
        raise NotImplementedError("This method must be overwritten in children classes")

    def get_dual_variables(self):
        """
        Output the list of dual variables.
        
        Returns:
            optimal_dual (list): numerical values of the dual variables (same ordering as that
                                 of _list_of_constraints_sent_to_solver).
            dual_residual (ndarray): dual variable corresponding to the main (primal) Gram matrix.

        """
        return self.dual_values, self.residual

    def get_primal_variables(self):
        """
        Output the optimal value of primal variables.
        
        Returns:
            optimal_G (ndarray): numerical Gram matrix of the PEP after solving.
            optimal_F (ndarray): numerical elements of F after solving.
            
        """
        return self.optimal_G, self.optimal_F

    def assign_dual_values(self):
        """
        Recover all dual variables and store them in associated :class:`Constraint` and :class:`PSDMatrix` objects.

        Returns:
            residual (ndarray): main dual PSD matrix (dual to the PSD constraint on the Gram matrix).

        Raises:
            TypeError if the attribute `_list_of_constraints_sent_to_solver` of this object
            is neither a :class:`Constraint` object, nor a :class:`PSDMatrix` one.

        """
        dual_values, residual = self._recover_dual_values()
        self.dual_values = dual_values
        self.residual = residual

        assert residual.shape == (Point.counter, Point.counter)

        for constraint_or_psd, dual_value in zip(self._list_of_constraints_sent_to_solver, dual_values[1:]):
            if isinstance(constraint_or_psd, Constraint):
                constraint_or_psd._dual_variable_value = dual_value
            elif isinstance(constraint_or_psd, PSDMatrix):
                assert dual_value.shape == constraint_or_psd.shape
                constraint_or_psd._dual_variable_value = dual_value
            else:
                raise TypeError("The list of constraints that are sent to CVXPY should contain only"
                                "\'Constraint\' objects of \'PSDMatrix\' objects."
                                "Got {}".format(type(constraint_or_psd)))

        return residual

    def _recover_dual_values(self):
        """
        Post-process the output of the solver and associate each constraint of the list
        _list_of_constraints_sent_to_solver to their corresponding numerical dual variables.
        
        Returns:
            dual_values (list)
            residual (np.array)

        """
        # Return dual_values, residual
        raise NotImplementedError("This method must be overwritten in children classes")

    def generate_problem(self, objective):
        """
        Instantiate an optimization model using the solver format, whose objective corresponds to a
        PEPit :class:`Expression` object.

        Args:
            objective (Expression): the objective function of the PEP (to be maximized).
        
        Returns:
            prob: the PEP in the solver's format.

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
            tol_dimension_reduction (float): tolerance on the objective for finding low-dimensional examples.
        
        """
        # Add constraint that tau <= wc_value + tol_dimension
        raise NotImplementedError("This method must be overwritten in children classes")

    def heuristic(self, weight):
        """
        Change the objective of the PEP, specifically for finding low-dimensional examples.
        We specify a matrix :math:`W` (weight), which will allow minimizing :math:`\\mathrm{Tr}(G\\,W)`.

        Args:
            weight (np.array): weights that will be used in the heuristic.
        
        """
        raise NotImplementedError("This method must be overwritten in children classes")
