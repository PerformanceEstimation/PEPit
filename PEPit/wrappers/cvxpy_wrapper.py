import importlib.util

from PEPit.wrapper import Wrapper
from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.constraint import Constraint
from PEPit.psd_matrix import PSDMatrix

from PEPit.tools.expressions_to_matrices import expression_to_matrices


class CvxpyWrapper(Wrapper):
    """
    A :class:`Cvxpy_wrapper` object interfaces PEPit with the `CVXPY <https://www.cvxpy.org/>`_ modelling language.

    This class overwrites the :class:`Wrapper` for CVXPY. In particular, it implements the methods:
    send_constraint_to_solver, send_lmi_constraint_to_solver, generate_problem, get_dual_variables,
    get_primal_variables, eval_constraint_dual_values, solve, prepare_heuristic, and heuristic.
    
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
        verbose (int): Level of information details to print
                       (Override the solver verbose parameter).

                       - 0: No verbose at all
                       - 1: PEPit information is printed but not solver's
                       - 2: Both PEPit and solver details are printed
        F (cvxpy.Variable): a 1D cvxpy.Variable that represents PEPit's Expressions.
        G (cvxpy.Variable): a 2D cvxpy.Variable that represents PEPit's Gram matrix.
        _list_of_solver_constraints (list of cvxpy.Constraint): the list of constraints of the problem in CVXPY format.

    """

    def __init__(self, verbose=1):
        """
        This function initialize all internal variables of the class. 
        
        Args:
            verbose (int): Level of information details to print
                           (Override the solver verbose parameter).

                           - 0: No verbose at all
                           - 1: PEPit information is printed but not solver's
                           - 2: Both PEPit and solver details are printed

        """
        super().__init__(verbose=verbose)

        # Initialize attributes
        self.F = None
        self.G = None
        self._list_of_solver_constraints = list()

    def set_main_variables(self):
        """
        Create base cvxpy variables and main cvxpy constraint: G >> 0.

        """
        import cvxpy as cp

        # Express the constraints from F, G and objective
        # Start with the main LMI condition
        self.F = cp.Variable((Expression.counter,))
        self.G = cp.Variable((Point.counter, Point.counter), symmetric=True)
        self._list_of_solver_constraints.append(self.G >> 0)

    def check_license(self):
        """
        Check that there is a valid available license for CVXPY.

        Returns:
            license presence (bool): no license needed: True

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
        import cvxpy as cp
        
        Gweights, Fweights, cons = expression_to_matrices(expression)
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
        import cvxpy as cp
        
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

    def _recover_dual_values(self):
        """
        Recover all dual variables from solver.

        Returns:
             dual_values (list): list of dual variables (floats) associated to _list_of_constraints_sent_to_solver
                                 (same ordering).
             residual (np.array): main dual PSD matrix (dual to the PSD constraint on the Gram matrix).

        Raises:
            TypeError if the attribute `_list_of_constraints_sent_to_solver` of this object
            is neither a :class:`Constraint` object, nor a :class:`PSDMatrix` one.

        """

        assert self._list_of_solver_constraints == self.prob.constraints
        dual_values_temp = [constraint.dual_value for constraint in self.prob.constraints]
        dual_values = list()

        # Store residual, dual value of the main lmi
        residual = dual_values_temp[0]
        dual_values.append(residual)
        assert residual.shape == (Point.counter, Point.counter)

        # Set counter
        counter = 1
        counter2 = 1  # number of dual variables (no artificial ones due to LMI)

        for constraint_or_psd in self._list_of_constraints_sent_to_solver:
            if isinstance(constraint_or_psd, Constraint):
                dual_values.append(dual_values_temp[counter])
                counter += 1
                counter2 += 1
            elif isinstance(constraint_or_psd, PSDMatrix):
                assert dual_values_temp[counter].shape == constraint_or_psd.shape
                dual_values.append(dual_values_temp[counter])
                counter += 1
                counter2 += 1
                size = constraint_or_psd.shape[0] * constraint_or_psd.shape[1]
                counter += size
            else:
                raise TypeError("The list of constraints that are sent to CVXPY should contain only"
                                "\'Constraint\' objects of \'PSDMatrix\' objects."
                                "Got {}".format(type(constraint_or_psd)))

        # Verify nothing is left
        assert len(dual_values) == counter2

        # Return the position of the reached performance metric
        return dual_values, residual

    def generate_problem(self, objective):
        """
        Instantiate an optimization model using the cvxpy format, whose objective corresponds to a
        PEPit :class:`Expression` object.

        Args:
            objective (Expression): the objective function of the PEP (to be maximized).
        
        Returns:
            prob (cvxpy.Problem): the PEP in cvxpy format.

        """
        import cvxpy as cp
        
        cvxpy_objective = self._expression_to_solver(objective)
        self.objective = cvxpy_objective
        self.prob = cp.Problem(objective=cp.Maximize(cvxpy_objective), constraints=self._list_of_solver_constraints)
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

        if "solver" in kwargs.keys():
            if kwargs["solver"] is None:
                kwargs["solver"] = "MOSEK"

        # Verify is CVXPY will try MOSEK first.
        if "solver" not in kwargs.keys() or kwargs["solver"] == "MOSEK":

            # If MOSEK is installed, CVXPY will run it.
            # We need to check the presence of a license and handle it in case there is no valid license.
            is_mosek_installed = importlib.util.find_spec("mosek")
            if is_mosek_installed:

                # Import mosek.
                import mosek

                # Create an environment.
                mosek_env = mosek.Env()

                # Grab the license if there is one.
                try:
                    mosek_env.checkoutlicense(mosek.feature.pton)
                except mosek.Error:
                    pass

                # Check validity of a potentially found license.
                if not mosek_env.expirylicenses() >= 0:

                    # In case the license is not valid, ask CVXPY to run SCS.
                    kwargs["solver"] = "SCS"

            else:
                # If mosek is not installed, ask CVXPY to run SCS.
                kwargs["solver"] = "SCS"

        # Solve the problem.
        self.prob.solve(**kwargs)

        # Store main information.
        self.solver_name = self.prob.solver_stats.solver_name
        self.optimal_G = self.G.value
        self.optimal_F = self.F.value

        # Return first information.
        return self.prob.status, self.solver_name, self.objective.value

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
        import cvxpy as cp

        obj = cp.sum(cp.multiply(self.G, weight))
        self.prob = cp.Problem(objective=cp.Minimize(obj), constraints=self._list_of_solver_constraints)
        return self.prob
