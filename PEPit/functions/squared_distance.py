from PEPit.point import Point
from PEPit.function import Function
from PEPit.functions.bregman_divergence import BregmanDivergenceTo


class _SquaredNorm(Function):
    """
    This class aims at being used only to define SquaredDistanceTo

    Notes:
        It must not count in Function.counter when being instantiated.
        It is not added to PEP.list_of_functions neither.

    """

    def __init__(self):
        super().__init__(is_leaf=True,
                         decomposition_dict=None,
                         reuse_gradient=True)

        # Make it not count in Function Counter
        self.counter = None
        Function.counter -= 1

    def add_class_constraints(self):
        """
        No constraint for this class.
        """
        pass

    def oracle(self, point):
        """
        Compute the known gradient and function value of the square norm:

        .. math:: \\nabla \\|x\\|^2 = 2 x

        Args:
            point (Point): any point.

        Returns:
            tuple: the gradient (:class:`Point`) and the function value (:class:`Expression`) computed on point.

        """

        # Compute gradient and function value
        gx = 2*point
        fx = point**2

        # Return gradient and function value
        return gx, fx


class SquaredDistanceTo(BregmanDivergenceTo):
    """
    The :class:`SquaredDistanceTo` class is a :class:`BregmanDivergenceTo` child class.
    It corresponds to the special case where the underlying function :math:`h` is the :class:`SquaredNorm`.
    It benefits from the closed form formula

    .. math::
        :nowrap:

            \\begin{eqnarray}
                d^2(x, x_0) & \\triangleq & \\| x - x_0 \\|^2, \\
                \\nabla d^2(x, x_0) & \\triangleq & 2(x - x_0).
            \\end{eqnarray}

    **References**:
    Definition and analyses of Bregman divergence can be found in [1].

    `[1] R. Dragomir, A. Taylor, A. d’Aspremont, J. Bolte (2021).
    Optimal Complexity and Certification of Bregman First-Order Methods.
    Mathematical Programming: 1-43.
    <https://arxiv.org/pdf/1911.08510.pdf>`_

    Squared distances are characterized by parameters and `x_0`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP, Point
        >>> from PEPit.functions import SquaredDistanceTo
        >>> problem = PEP()
        >>> x0 = Point()
        >>> func = problem.declare_function(function_class=SquaredDistanceTo, x0=x0)

    """

    def __init__(self,
                 x0,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
        """

        Args:
            x0 (Point): Point which the Bregman divergence is computing on.
            is_leaf (bool): True if self is defined from scratch.
                            False is self is defined as linear combination of leaf .
            decomposition_dict (dict): decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        """
        super().__init__(x0=x0,
                         h=_SquaredNorm(),
                         is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

    def fixed_point(self):
        """
        This routine outputs a fixed point of this function, that is :math:`x` such that :math:`x = \\nabla f(x)`.
        Since the gradient of self in :math:`x` is known to be :math:`2(x-x_0)`, the fixed point equation leads to

        .. math:: x = 2x_0

        Returns:
            x (Point): a fixed point of the differential of self.
            x (Point): \\nabla f(x) = x.
            fx (Expression): a function value.

        """

        # Compute the fixed point.
        x = 2 * self.x0

        # Compute the associated function value in x.
        fx = self.x0**2

        # Return the triplet verifying the fixed point equation.
        return x, x, fx
