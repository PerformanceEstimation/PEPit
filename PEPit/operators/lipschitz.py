import numpy as np
from PEPit.function import Function


class LipschitzOperator(Function):
    """
    The :class:`LipschitzOperator` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of Lipschitz continuous operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        L (float): Lipschitz parameter

    Cocoercive operators are characterized by the parameter :math:`L`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import LipschitzOperator
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=LipschitzOperator, L=1.)

    Notes:
        By setting L=1, we define a non-expansive operator.

        By setting L<1, we define a contracting operator.

    References:

        [1] M. Kirszbraun (1934).
        Uber die zusammenziehende und Lipschitzsche transformationen.
        Fundamenta Mathematicae, 22 (1934).

        [2] F.A. Valentine (1943).
        On the extension of a vector function so as to preserve a Lipschitz condition.
        Bulletin of the American Mathematical Society, 49 (2).

        [3] F.A. Valentine (1945).
        A Lipschitz condition preserving extension for a vector function.
        American Journal of Mathematics, 67(1).

        Discussions and appropriate pointers for the interpolation problem can be found in:
        `[4] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
        Operator splitting performance estimation: Tight contraction factors and optimal parameter selection.
        SIAM Journal on Optimization, 30(3), 2251-2271.
        <https://arxiv.org/pdf/1812.00146.pdf>`_

    """

    def __init__(self,
                 L,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """

        Args:
            L (float): Lipschitz continuity parameter.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Lipschitz continuous operators are necessarily continuous, hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )
        # Store L
        self.L = L

        if self.L == np.inf:
            print("\033[96m(PEPit) The class of L-Lipschitz operators with L == np.inf implies no constraint: \n"
                  "it contains all multi-valued mappings. This might imply issues in your code.\033[0m")

    def set_lipschitz_continuity_constraint_i_j(self,
                                                xi, gi, fi,
                                                xj, gj, fj,
                                                ):
        """
        Set Lipschitz continuity constraint for operators.

        """
        # Set constraint
        constraint = ((gi - gj) ** 2 - self.L ** 2 * (xi - xj) ** 2 <= 0)

        return constraint

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (Lipschitz operator),
        see [1, 2, 3] or e.g., [4, Fact 2].
        """

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="lipschitz_continuity",
                                                      set_class_constraint_i_j=
                                                      self.set_lipschitz_continuity_constraint_i_j,
                                                      symmetry=True,
                                                      )
