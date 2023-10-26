import unittest
import numpy as np

from PEPit.point import Point
from PEPit.expression import Expression

from PEPit.tools.expressions_to_matrices import expression_to_matrices, expression_to_sparse_matrices


class TestExpressionToMatrices(unittest.TestCase):

    def setUp(self):

        self.tearDown()

        self.point1 = Point()
        unused_point = Point()
        self.point2 = Point()
        unused_expr = Expression()
        self.expr = Expression()

        self.combined_expression = self.expr + self.point1 * self.point2 - 1

    def test_expression_to_matrices(self):

        # Run expression_to_matrices.
        Gweights, Fweights, cons = expression_to_matrices(self.combined_expression)

        # Compute expected outputs.
        Gweights_expected = np.array([[0., 0., 0.5], [0., 0., 0.], [0.5, 0., 0.]])
        Fweights_expected = np.array([0., 1.])
        cons_expected = -1

        # Compare the obtained outputs with the desired ones.
        G_error = np.sum((Gweights - Gweights_expected)**2)
        F_error = np.sum((Fweights - Fweights_expected)**2)
        cons_error = (cons - cons_expected)**2

        self.assertEqual(G_error, 0)
        self.assertEqual(F_error, 0)
        self.assertEqual(cons_error, 0)

    def test_expression_to_sparse_matrices(self):

        # Run expression_to_sparse_matrices.
        Gweights_indi, Gweights_indj, Gweights_val,\
            Fweights_ind, Fweights_val, cons_val = expression_to_sparse_matrices(self.combined_expression)

        # Compute expected outputs.
        Gweights_indi_expected = np.array([2])
        Gweights_indj_expected = np.array([0])
        Gweights_val_expected = np.array([0.5])
        Fweights_ind_expected = np.array([1])
        Fweights_val_expected = np.array([1])
        cons_val_expected = -1

        # Compare the obtained outputs with the desired ones.
        Gweights_indi_error = np.sum((Gweights_indi - Gweights_indi_expected) ** 2)
        Gweights_indj_error = np.sum((Gweights_indj - Gweights_indj_expected) ** 2)
        Gweights_val_error = (Gweights_val - Gweights_val_expected) ** 2
        Fweights_ind_error = (Fweights_ind - Fweights_ind_expected) ** 2
        Fweights_val_error = (Fweights_val - Fweights_val_expected) ** 2
        cons_val_error = (cons_val - cons_val_expected) ** 2

        self.assertEqual(Gweights_indi_error, 0)
        self.assertEqual(Gweights_indj_error, 0)
        self.assertEqual(Gweights_val_error, 0)
        self.assertEqual(Fweights_ind_error, 0)
        self.assertEqual(Fweights_val_error, 0)
        self.assertEqual(cons_val_error, 0)

    def tearDown(self):

        Expression.counter = 0
        Expression.list_of_leaf_expressions = list()

        Point.counter = 0
        Point.list_of_leaf_points = list()
