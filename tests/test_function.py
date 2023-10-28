import unittest

from PEPit import PEP
from PEPit.tools.dict_operations import prune_dict

from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.function import Function
from PEPit.functions.convex_function import ConvexFunction
from PEPit.operators.cocoercive_strongly_monotone import CocoerciveStronglyMonotoneOperator


class TestFunction(unittest.TestCase):

    def setUp(self):
        self.pep = PEP()

        self.func1 = Function(is_leaf=True, decomposition_dict=None, name="f1")
        self.func2 = ConvexFunction(is_leaf=True, decomposition_dict=None)
        self.func3 = self.pep.declare_function(CocoerciveStronglyMonotoneOperator, mu=.1, beta=1., name="f3")

        self.point = Point(is_leaf=True, decomposition_dict=None)

    def test_is_instance(self):

        self.assertIsInstance(self.func1, Function)
        self.assertIsInstance(self.func2, Function)
        self.assertIsInstance(self.func2, ConvexFunction)
        self.assertIsInstance(self.func3, Function)
        self.assertIsInstance(self.func3, CocoerciveStronglyMonotoneOperator)

    def test_name(self):

        self.assertIsNone(self.func2.get_name())
        self.assertEqual(self.func3.get_name(), "f3")

        self.func2.set_name("f2")

        self.assertEqual(self.func1.get_name(), "f1")
        self.assertEqual(self.func2.get_name(), "f2")

    def test_counter(self):

        composite_function = self.func1 + self.func2
        self.assertIs(self.func1.counter, 0)
        self.assertIs(self.func2.counter, 1)
        self.assertIs(self.func3.counter, 2)
        self.assertIs(composite_function.counter, None)
        self.assertIs(Function.counter, 3)

        new_function = Function(is_leaf=True, decomposition_dict=None)
        self.assertIs(new_function.counter, 3)
        self.assertIs(Function.counter, 4)

    def compute_linear_combination(self):

        new_function = - self.func1 + 2 * self.func2 - self.func2 / 5

        return new_function

    def test_linear_combination(self):

        new_function = self.compute_linear_combination()

        self.assertIsInstance(new_function, Function)
        self.assertEqual(new_function.decomposition_dict, {self.func1: -1, self.func2: 9 / 5})

    def test_callable(self):

        val = self.func1(point=self.point)

        self.assertEqual(len(self.func1.list_of_points), 1)

        triplet = self.func1.list_of_points[0]

        self.assertEqual(triplet[0], self.point)
        self.assertEqual(triplet[2], val)

    def test_oracle(self):

        new_function = self.compute_linear_combination()
        new_function.oracle(point=self.point)

        # On new_function
        self.assertEqual(len(new_function.list_of_points), 1)

        point, grad, val = new_function.list_of_points[0]
        self.assertIsInstance(point, Point)
        self.assertIsInstance(grad, Point)
        self.assertIsInstance(val, Expression)

        self.assertIs(point, self.point)
        self.assertTrue(grad._is_leaf)
        self.assertTrue(val._is_leaf)

        # On func1
        self.assertEqual(len(self.func1.list_of_points), 1)

        point1, grad1, val1 = self.func1.list_of_points[0]
        self.assertIsInstance(point1, Point)
        self.assertIsInstance(grad1, Point)
        self.assertIsInstance(val1, Expression)

        self.assertTrue(grad1._is_leaf)
        self.assertTrue(val1._is_leaf)

        # On func2
        self.assertEqual(len(self.func2.list_of_points), 1)

        point2, grad2, val2 = self.func2.list_of_points[0]
        self.assertIsInstance(point2, Point)
        self.assertIsInstance(grad2, Point)
        self.assertIsInstance(val2, Expression)

        self.assertFalse(grad2._is_leaf)
        self.assertFalse(val2._is_leaf)

        # Combination
        self.assertIs(point1, self.point)
        self.assertIs(point2, self.point)

        self.assertEqual((-grad1 + 9 * grad2 / 5).decomposition_dict, grad.decomposition_dict)
        self.assertEqual(prune_dict((-val1 + 9 * val2 / 5).decomposition_dict), val.decomposition_dict)

    def test_oracle_with_predetermined_values(self):

        # Compute composite function
        new_function = self.compute_linear_combination()

        # Compute oracle of each leaf function
        grad1, val1 = self.func1.oracle(point=self.point)
        grad2, val2 = self.func2.oracle(point=self.point)

        # Verifies the number of registered points
        self.assertEqual(len(self.func1.list_of_points), 1)
        self.assertEqual(len(self.func2.list_of_points), 1)

        # Compute oracle of composite function
        grad, val = new_function.oracle(point=self.point)

        # The value of composite must be determined, but the gradient must be new
        self.assertEqual(prune_dict(val.decomposition_dict), prune_dict((-val1 + 9/5*val2).decomposition_dict))
        self.assertNotEqual(prune_dict(grad.decomposition_dict), prune_dict((-grad1 + 9/5*grad2).decomposition_dict))

        # Verifies the number of registered points
        self.assertEqual(len(self.func1.list_of_points), 2)
        self.assertEqual(len(self.func2.list_of_points), 2)

        # Grab the new gradients and function values created for leaf functions
        other_grad1, other_val1 = self.func1.list_of_points[1][1:]
        other_grad2, other_val2 = self.func2.list_of_points[1][1:]

        # The function values must be the same, but not the gradients
        self.assertEqual(val1.decomposition_dict, other_val1.decomposition_dict)
        self.assertEqual(val2.decomposition_dict, other_val2.decomposition_dict)

        # The new gradients must match with the composite function gradient
        self.assertEqual(prune_dict(grad.decomposition_dict), prune_dict((-other_grad1 + 9/5*other_grad2).decomposition_dict))

    def test_oracle_with_predetermined_values_and_gradients(self):

        # First make self.func1 and self.func2 differentiable
        self.func1.reuse_gradient = True
        self.func2.reuse_gradient = True

        # Compute composite function
        new_function = self.compute_linear_combination()

        # Verify the composite function is differentiable as well
        self.assertTrue(new_function.reuse_gradient)

        # Compute oracle of each leaf function
        grad1, val1 = self.func1.oracle(point=self.point)
        grad2, val2 = self.func2.oracle(point=self.point)

        # Verifies the number of registered points
        self.assertEqual(len(self.func1.list_of_points), 1)
        self.assertEqual(len(self.func2.list_of_points), 1)

        # Compute oracle of composite function
        grad, val = new_function.oracle(point=self.point)

        # The value and gradient of composite must be determined
        self.assertEqual(prune_dict(val.decomposition_dict), prune_dict((-val1 + 9/5*val2).decomposition_dict))
        self.assertEqual(prune_dict(grad.decomposition_dict), prune_dict((-grad1 + 9/5*grad2).decomposition_dict))

        # Verifies the number of registered points
        # The latest must have not increased after calling for oracle on the composite function
        self.assertEqual(len(self.func1.list_of_points), 1)
        self.assertEqual(len(self.func2.list_of_points), 1)

    def test_stationary_point(self):

        # Compute composite function and define its stationary point
        new_function = self.compute_linear_combination()
        new_function.stationary_point()

        # On new_function
        self.assertEqual(len(new_function.list_of_points), 1)

        point, grad, val = new_function.list_of_points[0]
        self.assertIsInstance(point, Point)
        self.assertIsInstance(grad, Point)
        self.assertIsInstance(val, Expression)

        self.assertTrue(point._is_leaf)
        self.assertFalse(grad._is_leaf)
        self.assertTrue(val._is_leaf)

        self.assertEqual(grad.decomposition_dict, dict())
        self.assertEqual((grad ** 2).decomposition_dict, dict())

        self.assertEqual(len(new_function.list_of_stationary_points), 1)

        # On func1
        self.assertEqual(len(self.func1.list_of_points), 1)

        point1, grad1, val1 = self.func1.list_of_points[0]
        self.assertIsInstance(point1, Point)
        self.assertIsInstance(grad1, Point)
        self.assertIsInstance(val1, Expression)

        self.assertTrue(grad1._is_leaf)
        self.assertTrue(val1._is_leaf)

        self.assertEqual(len(self.func1.list_of_stationary_points), 0)

        # On func2
        self.assertEqual(len(self.func2.list_of_points), 1)

        point2, grad2, val2 = self.func2.list_of_points[0]
        self.assertIsInstance(point2, Point)
        self.assertIsInstance(grad2, Point)
        self.assertIsInstance(val2, Expression)

        self.assertFalse(grad2._is_leaf)
        self.assertFalse(val2._is_leaf)

        self.assertEqual(len(self.func2.list_of_stationary_points), 0)

        # Combination
        self.assertIs(point1, point)
        self.assertIs(point2, point)

        self.assertEqual((-grad1 + 9 * grad2 / 5).decomposition_dict, grad.decomposition_dict)
        self.assertEqual(prune_dict((-val1 + 9 * val2 / 5).decomposition_dict), val.decomposition_dict)

    def test_is_already_evaluated_on_points(self):
        new_function = self.compute_linear_combination()

        # Before adding a point
        self.assertEqual(new_function._is_already_evaluated_on_point(self.point), None)
        self.assertEqual(self.func1._is_already_evaluated_on_point(self.point), None)
        self.assertEqual(self.func2._is_already_evaluated_on_point(self.point), None)

        # Add a point
        new_function.oracle(point=self.point)

        # After adding a point
        self.assertEqual(new_function._is_already_evaluated_on_point(self.point), new_function.list_of_points[0][1:])
        self.assertEqual(self.func1._is_already_evaluated_on_point(self.point), self.func1.list_of_points[0][1:])
        self.assertEqual(self.func2._is_already_evaluated_on_point(self.point), self.func2.list_of_points[0][1:])

    def test_separate_leaf_functions_regarding_their_needs_on_points_non_differentiable(self):
        # Non differentiable case
        new_function = self.compute_linear_combination()
        point1 = Point(is_leaf=True, decomposition_dict=None)
        point2 = Point(is_leaf=True, decomposition_dict=None)
        new_function.oracle(point1)

        list_of_functions_which_need_nothing, list_of_functions_which_need_gradient_only, list_of_functions_which_need_gradient_and_function_value = new_function._separate_leaf_functions_regarding_their_need_on_point(point1)
        self.assertEqual(len(list_of_functions_which_need_nothing), 0)
        self.assertEqual(len(list_of_functions_which_need_gradient_only), 2)
        self.assertEqual(len(list_of_functions_which_need_gradient_and_function_value), 0)

        list_of_functions_which_need_nothing, list_of_functions_which_need_gradient_only, list_of_functions_which_need_gradient_and_function_value = new_function._separate_leaf_functions_regarding_their_need_on_point(
            point2)
        self.assertEqual(len(list_of_functions_which_need_nothing), 0)
        self.assertEqual(len(list_of_functions_which_need_gradient_only), 0)
        self.assertEqual(len(list_of_functions_which_need_gradient_and_function_value), 2)

    def test_separate_leaf_functions_regarding_their_needs_on_points_differentiable(self):
        # Non differentiable case
        new_function = Function(is_leaf=True, decomposition_dict=None, reuse_gradient=True)
        point1 = Point(is_leaf=True, decomposition_dict=None)
        point2 = Point(is_leaf=True, decomposition_dict=None)
        new_function.oracle(point1)

        list_of_functions_which_need_nothing, list_of_functions_which_need_gradient_only, list_of_functions_which_need_gradient_and_function_value = new_function._separate_leaf_functions_regarding_their_need_on_point(
            point1)
        self.assertEqual(len(list_of_functions_which_need_nothing), 1)
        self.assertEqual(len(list_of_functions_which_need_gradient_only), 0)
        self.assertEqual(len(list_of_functions_which_need_gradient_and_function_value), 0)

        list_of_functions_which_need_nothing, list_of_functions_which_need_gradient_only, list_of_functions_which_need_gradient_and_function_value = new_function._separate_leaf_functions_regarding_their_need_on_point(
            point2)
        self.assertEqual(len(list_of_functions_which_need_nothing), 0)
        self.assertEqual(len(list_of_functions_which_need_gradient_only), 0)
        self.assertEqual(len(list_of_functions_which_need_gradient_and_function_value), 1)
