import sys

import numpy as np

from PEPit.wrapper import Wrapper
from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.constraint import Constraint
from PEPit.psd_matrix import PSDMatrix

from PEPit.tools.expressions_to_matrices import expression_to_sparse_matrices


class MosekWrapper(Wrapper):
    """
    A :class:`MosekWrapper` object interfaces PEPit with the SDP solver `MOSEK <https://www.mosek.com/>`_.

    This class overwrites the :class:`Wrapper` for MOSEK. In particular, it implements the methods:
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
        _constraint_index_in_mosek (list of integer): indices corresponding to the list of constraints sent to MOSEK.
        _nb_pep_constraints_in_mosek (int): total number of scalar constraints sent to MOSEK.
        _list_of_psd_constraints_sent_to_solver (list): list of PSD constraints sent to MOSEK.
        _nb_pep_SDPconstraints_in_mosek (int): total number of PSD constraints sent to MOSEK.
        env: MOSEK environment
        task: Mosek task

    """

    def __init__(self, verbose=1):
        """
        This function initializes all internal variables of the class.
        
        Args:
            verbose (int): Level of information details to print
                           (Override the solver verbose parameter).

                           - 0: No verbose at all
                           - 1: PEPit information is printed but not solver's
                           - 2: Both PEPit and solver details are printed

        """
        super().__init__(verbose=verbose)

        # Initialize lists of constraints that are used to solve the SDP.
        # Those lists should not be updated by hand, only the solve method does update them.
        self._constraint_index_in_mosek = list()  # indices of the MOSEK constraints.
        self._nb_pep_constraints_in_mosek = 0
        self._list_of_psd_constraints_sent_to_solver = list()
        self._nb_pep_SDPconstraints_in_mosek = 0

        import mosek

        self.env = mosek.Env()
        self.task = self.env.Task()  # initiate MOSEK's task

        if self.verbose > 1:
            self.task.set_Stream(mosek.streamtype.log, self._streamprinter)

    def set_main_variables(self):
        """
        Initialize the main variables of the optimization problem and set the main constraint G >> 0.

        """
        import mosek

        # "Initialize" the Gram matrix
        self.task.appendbarvars([Point.counter])
        self._nb_pep_SDPconstraints_in_mosek += 1
        # "Initialize" function value variables (an additional variable "tau" is for handling the objective, hence +1)
        self.task.appendvars(Expression.counter + 1)

        inf = 1.0  # symbolical purposes
        for i in range(Expression.counter):
            self.task.putvarbound(i, mosek.boundkey.fr, -inf, +inf)  # no bounds on function values (nor on tau)

    def check_license(self):
        """
        Check that there is a valid available license for MOSEK.

        Returns:
            license (bool): is there a valid license?

        """
        import mosek

        try:
            self.env.checkoutlicense(mosek.feature.pton)
        except mosek.Error:
            return False

        return self.env.expirylicenses() >= 0  # number of days until license expires >= 0?

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
        import mosek

        # Sanity check
        assert isinstance(constraint, Constraint)
        inf = 1.0  # for symbolic purposes

        # Add constraint to the attribute _list_of_constraints_sent_to_mosek to keep track of
        # all the constraints that have been sent to mosek as well as the order.
        if track:
            self._list_of_constraints_sent_to_solver.append(constraint)
            self._nb_pep_constraints_in_mosek += 1

        # how many constraints in the task so far? This will be the constraint number
        nb_cons = self.task.getnumcon()

        # Add a mosek constraint via task
        self.task.appendcons(1)
        A_i, A_j, A_val, a_i, a_val, alpha_val = expression_to_sparse_matrices(constraint.expression)

        sym_A = self.task.appendsparsesymmat(Point.counter, A_i, A_j, A_val)
        self.task.putbaraij(nb_cons, 0, [sym_A], [1.0])
        self.task.putaijlist(nb_cons + np.zeros(a_i.shape, dtype=np.int8), a_i, a_val)

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
        import mosek

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
                A_i, A_j, A_val, a_i, a_val, alpha_val = expression_to_sparse_matrices(psd_matrix[i, j])
                # how many constraints in the task so far? This will be the constraint number
                nb_cons = self.task.getnumcon()
                # add a constraint in mosek
                self.task.appendcons(1)
                # in MOSEK format: matrices corresponding to the quadratic part of the expression
                sym_A1 = self.task.appendsparsesymmat(Point.counter, A_i, A_j,
                                                      A_val)  # 1/2 because we have to symmetrize the matrix!
                # in MOSEK format: this is the matrix which selects one entry of the matrix to be PSD
                sym_A2 = self.task.appendsparsesymmat(size, [max(i, j)], [min(i, j)], [
                    -.5 * (i != j) - 1 * (i == j)])  # 1/2 because we have to symmetrize the matrix!
                # fill the mosek (equality) constraint 
                self.task.putbaraij(nb_cons, 0, [sym_A1], [1.0])
                self.task.putbaraij(nb_cons, psd_matrix.counter + 1, [sym_A2], [1.0])
                self.task.putaijlist(nb_cons + np.zeros(a_i.shape, dtype=np.int8), a_i, a_val)
                self.task.putconbound(nb_cons, mosek.boundkey.fx, -alpha_val, -alpha_val)

        # Print a message if verbose mode activated
        if self.verbose > 0:
            print('\t\t Size of PSD matrix {}: {}x{}'.format(psd_counter + 1, *psd_matrix.shape))

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
        import mosek

        # MUST CHECK the nb of constraints, somehow
        scalar_dual_values = self.task.gety(mosek.soltype.itr)
        dual_values = list()

        counter_psd = 1
        counter_scalar = 0

        dual_values.append(-self._get_Gram_from_mosek(self.task.getbarsj(mosek.soltype.itr, 0), Point.counter))
        for constraint_or_psd in self._list_of_constraints_sent_to_solver:
            if isinstance(constraint_or_psd, Constraint):
                dual_values.append(scalar_dual_values[self._constraint_index_in_mosek[counter_scalar]])
                counter_scalar += 1

            elif isinstance(constraint_or_psd, PSDMatrix):
                dual_values.append(-self._get_Gram_from_mosek(self.task.getbarsj(mosek.soltype.itr, counter_psd),
                                                              constraint_or_psd.shape[0]))
                assert dual_values[-1].shape == constraint_or_psd.shape
                counter_psd += 1
            else:
                raise TypeError("The list of constraints that are sent to CVXPY should contain only"
                                "\'Constraint\' objects of \'PSDMatrix\' objects."
                                "Got {}".format(type(constraint_or_psd)))

        assert len(dual_values) - 1 == len(
            self._list_of_constraints_sent_to_solver)  # -1 because we added the dual corresponding to the Gram matrix
        residual = dual_values[0]

        return dual_values, residual

    def generate_problem(self, objective):
        """
        Instantiate an optimization model using the mosek format, whose objective corresponds to a
        PEPit :class:`Expression` object.

        Args:
            objective (Expression): the objective function of the PEP (to be maximized).
        
        Returns:
            prob (mosek.task): the PEP in mosek's format.

        """
        import mosek

        assert self.task.getmaxnumvar() == Expression.counter + 1
        self.objective = objective
        _, _, _, Fweights_ind, Fweights_val, _ = expression_to_sparse_matrices(objective)
        self.task.putclist(Fweights_ind,
                           Fweights_val)  # to be cleaned by calling _expression_to_sparse_matrices(objective)?
        # Input the objective sense (minimize/maximize)
        self.task.putobjsense(mosek.objsense.maximize)
        if self.verbose > 1:
            self.task.solutionsummary(mosek.streamtype.msg)
        return self.task

    def solve(self, **kwargs):
        """
        Solve the PEP.

        Args:
            kwargs (keywords, optional): solver specific arguments.
        
        Returns:
            status (string): status of the solution / problem.
            name (string): name of the solver.
            value (float): value of the performance metric after solving.
            problem (mosek task): solver-specific model of the PEP.
        
        """
        import mosek
        if "solver" in kwargs.keys():
            del kwargs["solver"]
        self.task.optimize(**kwargs)
        self.solver_name = "MOSEK"
        self.optimal_G = self._get_Gram_from_mosek(self.task.getbarxj(mosek.soltype.itr, 0), Point.counter)
        xx = self.task.getxx(mosek.soltype.itr)
        tau = xx[-2]
        self.optimal_F = xx
        problem_status = self.task.getprosta(mosek.soltype.itr)
        return problem_status, self.solver_name, tau

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
        import mosek

        self.task.putclist([Expression.counter - 1], [0.0])
        self.task.putobjsense(mosek.objsense.minimize)
        self.send_constraint_to_solver(self.objective >= wc_value - tol_dimension_reduction, track=False)

    def heuristic(self, weight):
        """
        Change the objective of the PEP, specifically for finding low-dimensional examples.
        We specify a matrix :math:`W` (weight), which will allow minimizing :math:`\\mathrm{Tr}(G\\,W)`.

        Args:
            weight (np.array): weights that will be used in the heuristic.
        
        """
        import mosek

        No_zero_ele = np.argwhere(np.tril(weight))
        W_i = No_zero_ele[:, 0]
        W_j = No_zero_ele[:, 1]
        W_val = weight[W_i, W_j]
        sym_W = self.task.appendsparsesymmat(Point.counter, W_i, W_j, W_val)
        self.task.putbarcj(0, [sym_W], [1.0])
        self.task.putobjsense(mosek.objsense.minimize)

    @staticmethod
    def _streamprinter(text):
        """
        Output summaries.

        Args:
            text (string): to be printed out.
        
        """
        sys.stdout.write(text)
        sys.stdout.flush()

    @staticmethod
    def _get_Gram_from_mosek(tril, size):
        """
        Compute a Gram matrix from mosek.

        Args:
            tril (numpy array): weights that will be used in the heuristic.
            size (int): weights that will be used in the heuristic.
        
        """
        # the primal solution for a semi-definite variable.
        # Only the lower triangular part of is returned because the matrix by construction is symmetric.
        # The format is that the columns are stored sequentially in the natural order.
        G = np.zeros((size, size))
        counter = 0
        for j in range(size):
            for i in range(size - j):
                G[j + i, j] = tril[counter]
                G[j, j + i] = tril[counter]
                counter += 1
        return G
