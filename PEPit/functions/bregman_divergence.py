from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.function import Function


class BregmanDivergenceTo(Function):
    """
    The :class:`BregmanDivergence` class is a special :class:`Function`.
    Its functions values and subgradients benefit from the closed form formulations

    .. math::
        :nowrap:

            \\begin{eqnarray}
                D_h(x; x_0) & \\triangleq & h(x) - h(x_0) - \\left< \\nabla h(x_0);\, x - x_0 \\right>, \\
                \\nabla D_h(x; x_0) & \\triangleq & \\nabla h(x) - \\nabla h(x_0),
            \\end{eqnarray}

    where :math:`h` is the underlying function we compute the Bregman divergence of.

    Warnings:
        The function :math:`h` is assumed to be closed strictly convex proper and continuously differentiable.
        (See [1, Definition 1])
        In particular, :math:`h` `subgradient` method must always return
        the same subgradient if call several times on the same point.
        Hint: set `reuse_gradient` attribute of :math:`h` to True.

    Hence, this class overwrites the `oracle`, `stationary_point` and `fixed_point` methods of :class:`Function`,
    and their is no complementary class constraint.

    Bregman divergences are characterized by parameters :math:`h` and `x_0`,
    hence can be instantiated as

    **References**:
    Definition and analyses of Bregman divergence can be found in [1].

    `[1] R. Dragomir, A. Taylor, A. d’Aspremont, J. Bolte (2021).
    Optimal Complexity and Certification of Bregman First-Order Methods.
    Mathematical Programming: 1-43.
    <https://arxiv.org/pdf/1911.08510.pdf>`_

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
                 reuse_gradient=True):
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

        # Warn if h.reuse_gradient if False
        if not self.h.reuse_gradient:
            print("\033[96mWarning: h must be continuously differentiable."
                  "The Bregman divergence computation may reuse the same subgradient of h on a given point"
                  " even if h's reuse_gradient attribute is set to False.\033[0m")

        # Store information that is commonly used in each call of this Function
        self.vector = self.h.subgradient(self.x0)
        self.scalar = self.h.value(self.x0) - self.vector * self.x0

    def add_class_constraints(self):
        """
        No constraint for this class.
        """
        pass

    def _is_already_evaluated_on_point(self, point):
        """
        Check whether this :class:`Function` is already evaluated on the :class:`Point` "point" or not.
        This method is used to determine whether we need to create a new gradient and a new function value
        when calling `add_point` or `oracle` of a function composed in part of self.
        The Bregman divergence, being defined by closed form formulation, it is already evaluated on every points.

        Args:
            point (Point): the point we want to check whether the function is evaluated on or not.

        Returns:
            tuple: return the tuple "gradient, function value" associated to "point".

        """

        return self.oracle(point=point)

    def oracle(self, point):
        """
        Return the gradient and the function value of self evaluated at `point`.
        The latest benefit from the following closed form formula.

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\nabla D_h(x; x_0) & = & \\nabla h(x) - \\nabla h(x_0) \\
                D_h(x; x_0) & = & h(x) - h(x_0) - \\left< \\nabla h(x_0) | x-x_0 \\right>
            \\end{eqnarray}

        Args:
            point (Point): any point.

        Returns:
            tuple: the gradient (:class:`Point`) and the function value (:class:`Expression`) computed on point.

        Notes:
            No point is stored in `list_of_points` attribute,
            the latest being useless as soon as this class is not defined by constraints.

        """

        # Computation of the gradient and function value on point.
        g = self.h.subgradient(point) - self.vector
        f = self.h.value(point) - self.vector * point - self.scalar

        # Return the computed gradient and function value.
        return g, f

    def stationary_point(self, return_gradient_and_function_value=False):
        """
        Return the unique stationary point :math:`x_0`, as well as its zero gradient and its function value.

        Notes:
            No point is stored in `list_of_points` attribute,
            the latest being useless as soon as this class is not defined by constraints.

        Args:
            return_gradient_and_function_value (bool): if True, return the triplet point (:class:`Point`),
                                                       subgradient (:class:`Point`),
                                                       function value (:class:`Expression`).
                                                       Otherwise, return only the point (:class:`Point`).

        Returns:
            Point or tuple: the optimal point :math:`x_0`.

        """

        # Call x0, null subgradient and null function value
        point = self.x0
        g = Point(is_leaf=False, decomposition_dict=dict())
        f = Expression(is_leaf=False, decomposition_dict=dict())

        # Return the required information
        if return_gradient_and_function_value:
            return point, g, f
        else:
            return point

    def fixed_point(self):
        """
        This routine outputs a fixed point of this function, that is :math:`x` such that :math:`x = \\nabla f(x)`.
        Since the gradient of self in :math:`x` is known, the fixed point equation is

        .. math:: \\nabla h(x) = x + \\nabla h(x_0)

        Returns:
            x (Point): a fixed point of the differential of self.
            x (Point): \\nabla f(x) = x.
            fx (Expression): a function value.

        """

        # Define a point.
        x = Point()

        # Compute the associated gradient of h in x.
        gx = x + self.vector

        # Instantiate a function value of h in x.
        hx = Expression()

        # Add the obtained triplet to self.h's list of points.
        self.h.add_point((x, gx, hx))

        # Compute function value of self in x.
        fx = hx - self.vector * x - self.scalar

        # Return the triplet verifying the fixed point equation.
        return x, x, fx
