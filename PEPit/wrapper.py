class Wrapper(object):
    """
    A :class:`Wrapper` object interfaces PEPit with an SDP solver (or modelling language).

    Warnings:
        This class must be overwritten by a child class that encodes all particularities of the solver.
        In particular, the methods: send_constraint_to_solver, send_lmi_constraint_to_solver, generate_problem,
        get_dual_variables, get_primal_variables, eval_constraint_dual_values, prepare_heuristic, and heuristic
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
    
    
    def check_license(self):
        """
        Check that there is a valid available license for the solver.

        Args:
            constraint (Constraint): a :class:`Constraint` object to be sent to the solver.

        Raises:
            ValueError if the attribute `equality_or_inequality` of the :class:`Constraint`
            is neither `equality`, nor `inequality`.

        Returns:
            license (bool): is there a valid license?

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
        #change the objective using weight and returns prob
        raise NotImplementedError("This method must be overwritten in children classes")
    
    def solve(self, **kwargs):
        """
        Change the objective of the PEP, specifically for finding low-dimensional examples.
        We specify a matrix :math:`W` (weight), which will allow minimizing :math:`\\mathrm{Tr}(G\\,W)`.

        Args:
            kwargs (keywords, optional): solver specific arguments.
        
        Returns:
            status (): solver-specific status.
            name (string): name of the solver.
            value (float): value of the performance metric after solving.
            problem (): solver-specific model of the PEP.
        
        """
        self.prob.solve(**kwargs)
        self.optimal_G, self.optimal_F = self.get_primal_variables()
        return self.prob.status, self.prob.solver_stats.solver_name, self.objective.value, self.prob
