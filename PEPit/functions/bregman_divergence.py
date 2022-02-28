# TODO:
# - if h is assumed to be differentiable, set reuse_gradient to True
#                                         + add warning!
#                                         + factorize vector and scalar
#                                         + don't bother with add_points
# - if h is not necessarily differentiable, create "new_grad" and "new_function_value" in Function
#                                           + overwrite those instead of oracle
#                                           + don't forget to update list_of_points


from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.function import Function


class BregmanDivergence(Function):
    """
    The :class:`BregmanDivergence` class is a special :class:`Function`.
    Its functions values and subgradients benefit from the closed form formulations:

    .. math::
        :nowrap:

            \\begin{eqnarray}
                D_h(x; x_0) & \\triangleq & h(x) - h(x_0) - \\left< \\nabla h(x_0);\, x - x_0 \\right>, \\
                \\nabla D_h(x; x_0) & \\triangleq & \\nabla h(x) - \\nabla h(x_0).
            \\end{eqnarray}

    Hence, this class overwrites the `oracle`, `stationary_point` and `fixed_point` methods of :class:`Function`,
    and their is no complementary class constraint.

    Bregman divergences are characterized by parameters :math:`h` and `x_0`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP, Point, Function
        >>> from PEPit.functions import BregmanDivergence
        >>> problem = PEP()
        >>> h = Function()
        >>> x0 = Point()
        >>> func = problem.declare_function(function_class=BregmanDivergence, param={'h': h, 'x0': x0})

    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """

        Args:
            param (dict): contains the values h and x0
            is_leaf (bool): True if self is defined from scratch.
                            False is self is defined as linear combination of leaf .
            decomposition_dict (dict): decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

        # Store the function h and the point x0
        self.h = param['h']
        self.x0 = param['x0']

    def add_class_constraints(self):
        """
        No constraint for this class.
        """
        pass

    def oracle(self, point):
        """
        Return a gradient (or a subgradient) and the function value of self evaluated at `point`.

        Args:
            point (Point): any point.

        Returns:
            tuple: a (sub)gradient (:class:`Point`) and a function value (:class:`Expression`).

        """

        vector = self.h.subgradient(self.x0)
        scalar = self.h.value(self.x0) - vector * self.x0

        g = self.h.subgradient(point) - vector
        f = self.h.value(point) - vector * point - scalar

        return g, f

    def stationary_point(self, return_gradient_and_function_value=False):

        """
        Create a new stationary point, as well as its zero subgradient and its function value.

        Args:
            return_gradient_and_function_value (bool): if True, return the triplet point (:class:`Point`),
                                                       subgradient (:class:`Point`),
                                                       function value (:class:`Expression`).
                                                       Otherwise, return only the point (:class:`Point`).

        Returns:
            Point or tuple: an optimal point

        """

        # Create a new point, null subgradient and new function value
        point = self.x0
        g = Point(is_leaf=False, decomposition_dict=dict())
        f = Expression(is_leaf=False, decomposition_dict=dict())

        # Add the triplet to the list of points of the function as well as to its list of stationary points
        self.add_point((point, g, f))

        # Return the required information
        if return_gradient_and_function_value:
            return point, g, f
        else:
            return point

    def fixed_point(self):

        """
        This routine outputs a fixed point of this function, that is :math:`x` such that :math:`x\\in\\partial f(x)`.
        If self is an operator :math:`A`, the fixed point is such that :math:`Ax = x`.

        Returns:
            x (Point): a fixed point of the differential of self.
            x (Point): \\nabla f(x) = x.
            fx (Expression): a function value (useful only if self is a function).

        """

        vector = self.h.subgradient(self.x0)
        scalar = self.h(self.x0) - vector * self.x0

        # Define a point and function value
        x = Point()
        hx = Expression()
        fx = hx - vector * x - scalar

        # Add triplet to self's list of points (by definition gx = x)
        self.add_point((x, x, fx))

        # Add triplet to self.h's list of points (by definition gx = x)
        self.h.add_point((x, x + vector, hx))

        # Return the aforementioned triplet
        return x, x, fx
