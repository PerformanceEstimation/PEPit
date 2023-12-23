from PEPit.function import Function


class NegativelyComonotoneOperator(Function):
    """
    The :class:`NegativelyComonotoneOperator` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing some necessary constraints of the class of negatively comonotone operators.

    Warnings:
        Those constraints might not be sufficient, thus the caracterized class might contain more operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        rho (float): comonotonicity parameter (>0)

    Negatively comonotone operators are characterized by the parameter :math:`\\rho`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import NegativelyComonotoneOperator
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=NegativelyComonotoneOperator, rho=1.)

    References:
        `[1] E. Gorbunov, A. Taylor, S. Horváth, G. Gidel (2023).
        Convergence of proximal point and extragradient-based methods beyond monotonicity:
        the case of negative comonotonicity.
        International Conference on Machine Learning.
        <https://proceedings.mlr.press/v202/gorbunov23a/gorbunov23a.pdf>`_

    """

    def __init__(self,
                 rho,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """

        Args:
            rho (float): The comonotonicity parameter (>0).
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
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

        # Store the beta parameter
        self.rho = rho

        if self.rho < 0:
            print("\033[96m(PEPit) The parameters rho is expected to be positive.\033[0m")

        if self.rho == 0:
            print("\033[96m(PEPit) The class of cocoercive operators is necessarily continuous. \n"
                  "To instantiate a monotone operator, please avoid using the class NegativelyComonotoneOperator\n"
                  "with rho == 0. Instead, please use the class Monotone.\033[0m")

    def set_negative_comonotonicity_constraint_i_j(self,
                                                   xi, gi, fi,
                                                   xj, gj, fj,
                                                   ):
        """
        Formulates the list of interpolation constraints for self (negatively comonotone operator).
        """
        # Interpolation conditions of negatively comonotone operator class
        constraint = ((gi - gj) * (xi - xj) + self.rho * (gi - gj) ** 2 >= 0)

        return constraint

    def add_class_constraints(self):
        """
        Add negative comonotonicity constraints.
        """
        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="negative_comonotonicity",
                                                      set_class_constraint_i_j=
                                                      self.set_negative_comonotonicity_constraint_i_j,
                                                      symmetry=True,
                                                      )
