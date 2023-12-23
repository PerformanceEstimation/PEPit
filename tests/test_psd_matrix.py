import unittest
import numpy as np

from PEPit.pep import PEP
from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.psd_matrix import PSDMatrix


class TestPSDMatrix(unittest.TestCase):

    def setUp(self):

        # Define problem, and base PEP objects
        self.problem = PEP()

        self.point = Point()

        self.expr1 = self.point ** 2
        self.expr2 = Expression()
        self.expr3 = Expression()

        # psd1 being a PSD matrix is equivalent to expr2 ** 2 <= expr1.
        psd1 = PSDMatrix([[self.expr1, self.expr2], [self.expr2, 1]], name="psd1")
        self.psd1 = self.problem.add_psd_matrix(psd1)

        # psd2 being a PSD matrix is equivalent to expr3 ** 2 <= expr1 * expr2.
        self.psd2 = self.problem.add_psd_matrix([[self.expr1, self.expr3], [self.expr3, self.expr2]], name="psd2")

        # In other words,
        #     the first lmi constraint implies:
        #         expr1 >= 0,
        #         expr2 is in [-expr1**(1/2), expr1**(1/2)],
        #     while the second one implies:
        #         expr2 >= 0,
        #         expr3 is in [-(expr1*expr2)**(1/2), (expr1*expr2)**(1/2)].

        # Adding an upper bound on expr1
        self.bound = np.abs(np.random.randn())
        self.problem.set_initial_condition(self.expr1 <= self.bound)
        # In summary:
        #     expr1 is in [0, bound],
        #     expr2 is in [0, expr1**(1/2)],
        #     expr3 is in [-(expr1*expr2)**(1/2), (expr1*expr2)**(1/2)].

        # Add performance metric
        self.problem.set_performance_metric(self.expr3)
        # The PEP must maximize expr3 <= (expr1*expr2)**(1/2) <= expr1 ** (3/4) <= bound ** (3/4).
        # The solution must therefore be expr3 = bound ** (3/4) when all the inequalities above are equalities, i.e.,
        #     expr1 = bound,
        #     expr2 = bound ** (1/2),
        #     expr3 = bound ** (3/4).
        # The primal variables then are
        #     [[bound, bound ** (1/2)], [bound ** (1/2), 1]],
        #     [[bound, bound ** (3/4)], [bound ** (3/4), bound ** (1/2)]].
        # And the dual variables are
        #     [[bound ** (-1/4)/4, - bound ** (1/4)/4], [- bound ** (1/4)/4, bound ** (3/4)/4]],
        #     [[bound ** (-1/4)/2, - 1/2], [- 1/2, bound ** (1/4)/2]].

        # Define a verbose
        self.verbose = 0

    def test_is_instance(self):

        self.assertIsInstance(self.problem, PEP)

        self.assertIsInstance(self.point, Point)

        self.assertIsInstance(self.expr1, Expression)
        self.assertIsInstance(self.expr2, Expression)
        self.assertIsInstance(self.expr3, Expression)

        self.assertIsInstance(self.psd1, PSDMatrix)
        self.assertIsInstance(self.psd2, PSDMatrix)

    def test_counter(self):

        self.assertEqual(self.problem.counter, 0)
        self.assertEqual(PEP.counter, 1)

        self.assertEqual(self.point.counter, 0)
        self.assertEqual(Point.counter, 1)

        self.assertIsNone(self.expr1.counter)
        self.assertEqual(self.expr2.counter, 0)
        self.assertEqual(self.expr3.counter, 1)
        self.assertEqual(Expression.counter, 2)

        self.assertEqual(self.psd1.counter, 0)
        self.assertEqual(self.psd2.counter, 1)
        self.assertEqual(PSDMatrix.counter, 2)

    def test_name(self):

        self.assertEqual(self.psd1.get_name(), "psd1")
        self.assertEqual(self.psd2.get_name(), "psd2")

        self.psd1.set_name("new_name")
        self.assertEqual(self.psd1.get_name(), "new_name")

    def test_getitem(self):

        self.assertIs(self.psd1[0, 0], self.expr1)
        self.assertIs(self.psd2[0, 1], self.expr3)

    def test_eval(self):
        
        # The PEP has not been solved yet, so no value is accessible.
        self.assertRaises(ValueError, self.psd1.eval)
        self.assertRaises(ValueError, self.psd2.eval)

        # Solve the problem.
        self.problem.solve(verbose=self.verbose)
        
        # Now we can have access to the optimal values of the PSD matrices.
        optimal_psd1 = [[self.bound, self.bound ** (1/2)], [self.bound ** (1/2), 1]]
        self.assertAlmostEqual(np.sum(self.psd1.eval() - optimal_psd1)**2, 0)

        optimal_psd2 = [[self.bound, self.bound ** (3/4)], [self.bound ** (3/4), self.bound ** (1/2)]]
        self.assertAlmostEqual(np.sum(self.psd2.eval() - optimal_psd2)**2, 0)
    
    def test_eval_dual(self):
        
        # The PEP has not been solved yet, so no dual value is accessible.
        self.assertRaises(ValueError, self.psd1.eval_dual)
        self.assertRaises(ValueError, self.psd2.eval_dual)

        # Solve the problem.
        self.problem.solve(verbose=self.verbose)

        # Now we can have access to the optimal dual values of the LMI constraints.
        optimal_dual_lmi1 = [[self.bound ** (-1/4)/4, - self.bound ** (1/4)/4], [- self.bound ** (1/4)/4, self.bound ** (3/4)/4]]
        self.assertAlmostEqual(np.sum(self.psd1.eval_dual() - optimal_dual_lmi1)**2, 0)

        optimal_dual_lmi2 = [[self.bound ** (-1/4)/2, - 1/2], [- 1/2, self.bound ** (1/4)/2]]
        self.assertAlmostEqual(np.sum(self.psd2.eval_dual() - optimal_dual_lmi2)**2, 0)
