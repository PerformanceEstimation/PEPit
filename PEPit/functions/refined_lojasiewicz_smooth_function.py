from PEPit.function import Function


class Refined_LojasiewiczSmoothFunction(Function):
    def __init__(self,
                 L,
                 mu,
                 alpha = None,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )
        assert L >= 0
        assert mu >= 0
        assert L >= mu
        if alpha != None:
        	assert alpha >= 0
        	assert alpha <= 2*mu/(2*L+mu)
        self.mu = mu
        self.L = L
        self.alpha = alpha
    def last_call_before_problem_formulation(self):
        if self.list_of_stationary_points == list():
            self.stationary_point()
        
    def set_LojaSimple(self,
                       xi, gi, fi,
                       xj, gj, fj,
                      ):
        
        constraint = (fi - fj <= gi**2 / 2 / self.mu)

        return constraint
        
    def set_LowerSimple(self,
                        xi, gi, fi,
                        xj, gj, fj,
                       ):
        
        constraint = (fi - fj >= gi**2 / 2 / self.L)

        return constraint
        
    def set_CombinedSmoothnessLojasiewicz(self,
                             xi, gi, fi,
                             xj, gj, fj,
                            ):
        const = (self.L+self.mu)*(1-self.alpha)**2 / ((self.L+self.mu)*(1-self.alpha)**2-(self.L-self.mu))

        constraint = (fi - fj >= 1/4 * (gi + gj) * (xi - xj) + 1 / (4 * self.L) * (gj - gi) ** 2 - self.L/4 * (xj - xi)**2
                      + self.alpha / ( 1 - self.alpha ) * ( (fj + gj**2 / 2/self.L) - self.L/4 * const * (xj - xi + (gi+gj)/self.L)**2 ) )

        return constraint
        
    def set_SmoothnessSimple(self,
                             xi, gi, fi,
                             xj, gj, fj,
                            ):
        constraint = (fi - fj >= 1/4 * (gi + gj) * (xi - xj) + 1 / (4 * self.L) * (gj - gi) ** 2 - self.L/4 * (xj - xi)**2 )

        return constraint
    
    def add_class_constraints(self):

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_stationary_points,
                                                      constraint_name="basic_Lojasiewicz",
                                                      set_class_constraint_i_j=self.set_LojaSimple,
                                                      )

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_stationary_points,
                                                      constraint_name="lower_bound",
                                                      set_class_constraint_i_j=self.set_LowerSimple,
                                                      )

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="smoothness",
                                                      set_class_constraint_i_j=self.set_SmoothnessSimple,
                                                      )

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="combined_smoothness_lojasiewicz",
                                                      set_class_constraint_i_j=self.set_CombinedSmoothnessLojasiewicz,
                                                      )

