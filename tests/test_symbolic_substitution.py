import unittest
from fractions import Fraction

from PEPit import PEP, SymbolicScalar
from PEPit.functions import SmoothStronglyConvexQuadraticFunction


def build_gd_quadratic_problem(mu, L, gamma, n=1):
    problem = PEP()
    func = problem.declare_function(SmoothStronglyConvexQuadraticFunction, mu=mu, L=L)

    xs = func.stationary_point()
    fs = func(xs)
    x0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    x = x0
    for _ in range(n):
        x = x - gamma * func.gradient(x)

    problem.set_performance_metric(func(x) - fs)
    return problem


class TestSymbolicSubstitution(unittest.TestCase):

    def test_symbolic_scalar_rejects_float_constructor(self):
        with self.assertRaises(TypeError):
            SymbolicScalar(0.3, "mu")

    def test_symbolic_substitution_matches_numeric_solve(self):
        mu_num = 0.3
        L_num = 3.0
        gamma_num = 1 / 3

        numeric_problem = build_gd_quadratic_problem(
            mu=mu_num, L=L_num, gamma=gamma_num, n=1
        )
        numeric_tau = numeric_problem.solve(verbose=0, solver="SCS")

        mu = SymbolicScalar(Fraction(3, 10), "mu")
        L = SymbolicScalar(3, "L")
        gamma = SymbolicScalar(Fraction(1, 3), "gamma")
        symbolic_problem = build_gd_quadratic_problem(mu=mu, L=L, gamma=gamma, n=1)
        symbolic_tau = symbolic_problem.solve(verbose=0, solver="SCS")

        self.assertAlmostEqual(numeric_tau, symbolic_tau, places=5)

    def test_symbolic_defaults_are_enough_without_solve_substitution(self):
        mu_num = 0.3
        L_num = 3.0
        gamma_num = 1 / 3

        numeric_problem = build_gd_quadratic_problem(
            mu=mu_num, L=L_num, gamma=gamma_num, n=1
        )
        numeric_tau = numeric_problem.solve(verbose=0, solver="SCS")

        mu = SymbolicScalar(Fraction(3, 10), "mu")
        L = SymbolicScalar(3, "L")
        gamma = SymbolicScalar(Fraction(1, 3), "gamma")
        symbolic_problem = build_gd_quadratic_problem(mu=mu, L=L, gamma=gamma, n=1)
        symbolic_tau = symbolic_problem.solve(verbose=0, solver="SCS")

        self.assertAlmostEqual(numeric_tau, symbolic_tau, places=5)


if __name__ == "__main__":
    unittest.main()
