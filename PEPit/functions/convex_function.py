import numpy as np
import pandas as pd
from PEPit.function import Function


class ConvexFunction(Function):
    """
    The :class:`ConvexFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of convex, closed and proper (CCP) functions (i.e., convex
    functions whose epigraphs are non-empty closed sets).

    General CCP functions are not characterized by any parameter, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import ConvexFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=ConvexFunction)

    """

    def __init__(self,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False,
                 name=None):
        """

        Args:
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient,
                         name=name,
                         )

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (CCP function).
        """

        for point_i in self.list_of_points:

            xi, gi, fi = point_i

            for point_j in self.list_of_points:

                xj, gj, fj = point_j

                if point_i != point_j:

                    # Interpolation conditions of convex functions class
                    constraint = (fi - fj >= gj * (xi - xj))
                    if None not in {self.name, xi.name, xj.name}:
                        constraint.set_name("IC_{}({}, {})".format(self.name, xi.name, xj.name))
                    self.list_of_class_constraints.append(constraint)

    def display_class_constraint_duals(self):

        n = len(self.list_of_points)
        list_of_duals = [constraint.eval_dual() for constraint in self.list_of_class_constraints]
        assert len(list_of_duals) == n*(n-1)
        complete_list_of_duals = [0]
        for i in range(n-1):
            complete_list_of_duals += list_of_duals[i*n: (i+1)*n]
            complete_list_of_duals += [0]
        tab_of_duals = pd.array(complete_list_of_duals)
        print(tab_of_duals)
