from PEPit.function import Function
from PEPit.expression import Expression
from PEPit import PSDMatrix
import numpy as np

class ExpertRefined_LojasiewiczSmoothFunction(Function):
    def __init__(self,
                 L,
                 mu,
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
        
        self.mu = mu
        self.L = L
        
    def last_call_before_problem_formulation(self):
        """
        FXXX

        """
        if self.list_of_stationary_points == list():
            self.stationary_point()
            
        nb_pts = len(self.list_of_points)
        preallocate = nb_pts * (nb_pts-1)
        self.M13 = np.ndarray((preallocate,),dtype=Expression)
        self.M14 = np.ndarray((preallocate,),dtype=Expression)
        self.M24 = np.ndarray((preallocate,),dtype=Expression)
        for i in range(preallocate):
            self.M13[i] = Expression()
            self.M14[i] = Expression()
            self.M24[i] = Expression()
        
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

        # Browse list of points and create necessary constraints for interpolation [XXX, Lemma XX]
        counter = 0
        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i
            xi_id = xi.get_name()
            if xi_id is None:
                xi_id = "Point_{}".format(i)

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j
                xj_id = xj.get_name()
                if xj_id is None:
                    xj_id = "Point_{}".format(j)
                
                if not (point_i == point_j):
                
                    A = -fi + fj + 1/2 * ( gi + gj ) * ( xi - xj ) + 1/4/self.L *  ( gi - gj )**2 - self.L/4 * ( xi - xj )**2 
                    B = (self.L + self.mu) * ( fi - 1/2/self.L * gi**2 )
                    C = (self.L - self.mu) * ( fj - 1/2/self.L * gj**2 )
                    D = 2 * self.mu * ( B - C - ( self.L + 3 * self.mu ) * A ) / ( 2 * self.L + self.mu)
                    E = 4 * self.mu**2 * (( self.L + self.mu) * A + B ) / ( 2 * self.L + self.mu) **2
                    F = - 2 * self.mu * A - D - E - 8 * self.mu**3 * B / ( 2 * self.L + self.mu )**3
                    M22 = - 6 * self.mu * A - D - 2 * self.M13[counter]
                    M33 = - 6 * self.mu * A - 2 * D + E - 2 * self.M24[counter] 
                    T = np.array([[-2*self.mu*A, 0, self.M13[counter], self.M14[counter]], [0,M22, -self.M14[counter], self.M24[counter]], [self.M13[counter],-self.M14[counter],M33,0],[self.M14[counter],self.M24[counter],0,F]], dtype=Expression)
                    psd_matrix = PSDMatrix(matrix_of_expressions=T)
                    self.list_of_class_psd.append(psd_matrix)
                    counter += 1 
