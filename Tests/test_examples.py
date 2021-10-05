import unittest

from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.function import Function

from Examples.gradient_descent_str_cvx_smooth import compute_rate_and_proof_n_steps_gd_on_str_cvx_smooth


class TestExamples(unittest.TestCase):

    def setUp(self):
        self.n = 6
        self.mu = .1
        self.L = 1

    def test_gd_str_cvx_smooth(self):
        rate, _ = compute_rate_and_proof_n_steps_gd_on_str_cvx_smooth(mu=self.mu,
                                                                      L=self.L,
                                                                      n=self.n,
                                                                      )

        expected_rate = ((self.L - self.mu) / (self.L + self.mu)) ** (2 * self.n)

        self.assertAlmostEqual(rate, expected_rate, delta=10 ** -4 * expected_rate)

    def test_gd_str_cvx_smooth_dual_variable(self):
        rate, dual_values = compute_rate_and_proof_n_steps_gd_on_str_cvx_smooth(mu=self.mu,
                                                                                L=self.L,
                                                                                n=1,
                                                                                )

        expected_dual = 4 * (self.L - self.mu) / (self.L + self.mu) ** 2

        self.assertAlmostEqual(dual_values[0], expected_dual, delta=10 ** -4 * expected_dual)
        self.assertAlmostEqual(dual_values[1], expected_dual, delta=10 ** -4 * expected_dual)

    def tearDown(self):
        Point.counter = 0
        Expression.counter = 0
        Function.counter = 0
