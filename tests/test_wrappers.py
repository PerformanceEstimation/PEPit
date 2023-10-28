import unittest
import numpy as np

from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction

from PEPit.tools.dict_operations import symmetrize_dict, prune_dict


class TestWrapperCVXPY(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wrapper = "cvxpy"

    def setUp(self):
        # Smooth strongly convex gradient descent set up
        self.L = 1.
        self.mu = 0.1
        self.gamma = 1 / self.L

        # Instantiate PEP
        self.problem = PEP()

        # Declare a strongly convex smooth function
        self.func = self.problem.declare_function(SmoothStronglyConvexFunction, mu=self.mu, L=self.L)

        # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
        self.xs = self.func.stationary_point()

        # Then define the starting point x0 of the algorithm
        self.x0 = self.problem.set_initial_point()

        # Set the initial constraint that is the distance between x0 and x^*
        self.problem.set_initial_condition((self.x0 - self.xs) ** 2 <= 1)

        # Run n steps of the GD method
        self.x1 = self.x0 - self.gamma * self.func.gradient(self.x0)

        # Set an expression lowering the norm of x1-xs.
        expr = Expression()
        self.problem.add_psd_matrix([[(self.x1 - self.xs) ** 2, expr], [expr, 1]])

        # Set the performance metric to the function values accuracy
        # Since we maximize expr, its value will exactly be the one of the norm of x1-xs.
        self.problem.set_performance_metric(expr)

        # Compute theoretical rate of the above problem
        self.theoretical_tau = max((1 - self.mu * self.gamma) ** 2, (1 - self.L * self.gamma) ** 2)

        # Define a verbose for all tests
        self.verbose = 0

    def test_dimension_reduction(self):

        # Compute pepit_tau very basically.
        pepit_tau = self.problem.solve(verbose=self.verbose, wrapper=self.wrapper)

        # Compute pepit_tau very basically with dimension_reduction_heuristic off and verify all is fine.
        pepit_tau2 = self.problem.solve(verbose=self.verbose, dimension_reduction_heuristic=None, wrapper=self.wrapper)
        self.assertAlmostEqual(pepit_tau2, pepit_tau, delta=10 ** -2)

        # Verify that, even with dimension reduction (using trace heuristic),
        # the solve method returns the worst-case performance, not the chosen heuristic value.
        pepit_tau3 = self.problem.solve(verbose=self.verbose, dimension_reduction_heuristic="trace", wrapper=self.wrapper)
        self.assertAlmostEqual(pepit_tau3, pepit_tau, delta=10 ** -2)

        # Verify that, even with dimension reduction (using 2 steps of local regularization of the log det heuristic),
        # the solve method returns the worst-case performance, not the chosen heuristic value.
        pepit_tau4 = self.problem.solve(verbose=self.verbose, dimension_reduction_heuristic="logdet2", wrapper=self.wrapper)
        self.assertAlmostEqual(pepit_tau4, pepit_tau, delta=10 ** -2)

    def test_track_constraints_sent_to_solver(self):

        # Run problem to send constraints to wrapper who sends it to solver
        self.problem.solve(wrapper=self.wrapper, verbose=self.verbose)

        # The wrapper should have sent 5 constraints to the solver: 1 initial, 2 class interpolation, 1 PSD
        # and 1 for the objective.
        self.assertEqual(len(self.problem.wrapper._list_of_constraints_sent_to_solver), 5)

    def test_recover_dual_values(self):

        # Run problem and grab the dual values back.
        self.problem.solve(wrapper=self.wrapper, verbose=self.verbose)
        dual_values, residual = self.problem.wrapper._recover_dual_values()

        # The wrapper should have sent 5 constraints to the solver: 1 initial, 2 class interpolation, 1 PSD
        # and 1 for the objective.
        # Then there must be 5 dual variables and a residual of the size of G,
        # that is the number of leaf points: 3 (xs, x0, g0).
        self.assertEqual(len(dual_values), 6)
        self.assertIs(dual_values[0], residual)
        self.assertEqual(residual.shape, (3, 3))

    def test_dual_sign_in_equality_constraints(self):

        # The equality 1 = some_expression does not lead to the same constraint's expression based on the class of 1.
        expr = Expression(is_leaf=False, decomposition_dict={1: 1})

        # Browse constraints and store reconstituted element of proof
        elements_of_proof = list()
        for constraint in [(self.x0 - self.xs) ** 2 <= expr,
                           expr >= (self.x0 - self.xs) ** 2,
                           (self.x0 - self.xs) ** 2 == expr,
                           expr == (self.x0 - self.xs) ** 2,
                           (self.x0 - self.xs) ** 2 <= 1,
                           1 >= (self.x0 - self.xs) ** 2,
                           (self.x0 - self.xs) ** 2 == 1,
                           1 == (self.x0 - self.xs) ** 2,
                           ]:
            self.problem.list_of_constraints = [constraint]
            self.problem.solve(verbose=self.verbose, wrapper=self.wrapper)
            elements_of_proof.append(constraint.eval_dual() * constraint.expression)

        # Test whether all elements of proofs are identical
        comparison_dict = elements_of_proof[0].decomposition_dict
        for element in elements_of_proof[1:]:
            for key, value in comparison_dict.items():
                self.assertAlmostEqual(value, element.decomposition_dict[key], delta=10**-5)

    def test_proof_consistency(self):

        # Solve the problem
        self.problem.solve(verbose=self.verbose, wrapper=self.wrapper)

        # - <Gram, residual> <= 0
        constraints_combination = -np.dot(Point.list_of_leaf_points,
                                          np.dot(self.problem.residual,
                                                 Point.list_of_leaf_points))

        # LMI constraints
        # Dual >= 0
        if self.problem._list_of_psd_sent_to_wrapper:
            # - <psd_matrix, lmi_dual> <= 0
            for psd_matrix in self.problem._list_of_psd_sent_to_wrapper:
                constraints_combination -= np.sum(psd_matrix.eval_dual() * psd_matrix.matrix_of_expressions)

        # Scalar constraints
        # + <expression, dual> <= 0
        for constraint in self.problem._list_of_constraints_sent_to_wrapper:
            constraints_combination += constraint.eval_dual() * constraint.expression

        # Proof reconstruction
        # At this stage, constraints_combination must be equal to "objective - tau"
        # which constitutes the proof as it has to be non-positive.
        # Compute an expression that should be exactly equal to the constant tau.
        dual_objective_expression = self.problem.objective - constraints_combination
        # Operation over the decomposition dict of dual_objective_expression
        dual_objective_expression_decomposition_dict = prune_dict(
            symmetrize_dict(dual_objective_expression.decomposition_dict)
        )
        # Remove the actual dual_objective from its dict
        del dual_objective_expression_decomposition_dict[1]
        # Compute the remaining terms, that should be small and only due to numerical stability errors
        remaining_terms = np.sum(np.abs([value for key, value in dual_objective_expression_decomposition_dict.items()]))

        # Check whether the proof is complete or not.
        # There should be no term left.
        self.assertAlmostEqual(remaining_terms, 0, delta=10**-5)


class TestWrapperMOSEK(TestWrapperCVXPY):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wrapper = "mosek"
