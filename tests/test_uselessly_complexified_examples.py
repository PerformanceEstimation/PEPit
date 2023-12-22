import unittest

from PEPit.examples.unconstrained_convex_minimization import wc_proximal_point
from PEPit.examples.composite_convex_minimization import wc_proximal_gradient
from PEPit.examples.unconstrained_convex_minimization import wc_gradient_exact_line_search
from PEPit.examples.unconstrained_convex_minimization import wc_inexact_gradient_exact_line_search
from PEPit.examples.stochastic_and_randomized_convex_minimization import \
    wc_randomized_coordinate_descent_smooth_strongly_convex
from PEPit.examples.stochastic_and_randomized_convex_minimization import wc_randomized_coordinate_descent_smooth_convex

from tests.additional_complexified_examples_tests import wc_proximal_gradient_complexified
from tests.additional_complexified_examples_tests import wc_proximal_gradient_complexified2
from tests.additional_complexified_examples_tests import wc_proximal_point_complexified
from tests.additional_complexified_examples_tests import wc_proximal_point_complexified2
from tests.additional_complexified_examples_tests import wc_proximal_point_complexified3
from tests.additional_complexified_examples_tests import wc_gradient_exact_line_search_complexified
from tests.additional_complexified_examples_tests import wc_inexact_gradient_exact_line_search_complexified
from tests.additional_complexified_examples_tests import wc_inexact_gradient_exact_line_search_complexified2
from tests.additional_complexified_examples_tests import wc_inexact_gradient_exact_line_search_complexified3
from tests.additional_complexified_examples_tests import \
    wc_randomized_coordinate_descent_smooth_strongly_convex_complexified
from tests.additional_complexified_examples_tests import wc_randomized_coordinate_descent_smooth_convex_complexified
from tests.additional_complexified_examples_tests import wc_gradient_descent_useless_blocks
from tests.additional_complexified_examples_tests import wc_gradient_descent_blocks


class TestExamples(unittest.TestCase):

    def setUp(self):
        self.n = 6
        self.mu = .1
        self.L = 1
        self.relative_precision = 10 ** -3

        self.verbose = -1

    def test_PGD_modified(self):
        L, mu, gamma, n = 1, 0.1, 1, 2

        wc, theory = wc_proximal_gradient_complexified(L, mu, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_PGD_vs_PGD_modified(self):
        L, mu, gamma, n = 1, 0.1, 1, 2

        wc_modified, theory = wc_proximal_gradient_complexified(L, mu, gamma, n, verbose=self.verbose)
        wc, theory = wc_proximal_gradient(L, mu, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=self.relative_precision * theory)

    def test_PGD_vs_PGD_modified2(self):
        L, mu, gamma, n = 1, 0.15, 1, 3

        wc_modified, theory = wc_proximal_gradient_complexified2(L, mu, gamma, n, verbose=self.verbose)
        wc, theory = wc_proximal_gradient(L, mu, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=self.relative_precision * theory)

    def test_PPA_modified(self):
        n, gamma = 2, 1.3

        wc, theory = wc_proximal_point_complexified(gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_PPA_modified2(self):
        n, gamma = 5, 1.3

        wc, theory = wc_proximal_point_complexified3(gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_PPA_vs_PPA_modified(self):
        n, gamma = 2, 1.1

        wc_modified, theory = wc_proximal_point_complexified(gamma, n, verbose=self.verbose)
        wc, theory = wc_proximal_point(gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=self.relative_precision * theory)

    def test_PPA_vs_PPA_modified2(self):
        n, gamma = 3, 2.1

        wc_modified, theory = wc_proximal_point_complexified2(gamma, n, verbose=self.verbose)
        wc, theory = wc_proximal_point(gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=self.relative_precision * theory)

    def test_PPA_vs_PPA_modified3(self):
        n, gamma = 4, 3.3

        wc_modified, theory = wc_proximal_point_complexified3(gamma, n, verbose=self.verbose)
        wc, theory = wc_proximal_point(gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=self.relative_precision * theory)

    def test_ELS_modified(self):
        L, mu, n = 3, .3, 3

        wc, theory = wc_gradient_exact_line_search_complexified(L=L, mu=mu, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inexact_ELS_modified(self):
        L, mu, epsilon, n = 2, .05, .2, 2

        wc, theory = wc_inexact_gradient_exact_line_search_complexified(L=L, mu=mu, epsilon=epsilon, n=n,
                                                                        verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inexact_ELS_modified2(self):
        L, mu, epsilon, n = 2, .05, .2, 2

        wc, theory = wc_inexact_gradient_exact_line_search_complexified2(L=L, mu=mu, epsilon=epsilon, n=n,
                                                                         verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inexact_ELS_modified3(self):
        L, mu, epsilon, n = 2, .05, .2, 2

        wc, theory = wc_inexact_gradient_exact_line_search_complexified3(L=L, mu=mu, epsilon=epsilon, n=n,
                                                                         verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_ELS_vs_ELS_modified(self):
        L, mu, n = 1.5, .12, 3

        wc_modified, theory = wc_gradient_exact_line_search_complexified(L=L, mu=mu, n=n, verbose=self.verbose)
        wc, theory = wc_gradient_exact_line_search(L=L, mu=mu, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=self.relative_precision * theory)

    def test_inexact_ELS_vs_ELS_modified(self):
        L, mu, epsilon, n = 2.3, .23, .2, 2

        wc_modified, theory = wc_inexact_gradient_exact_line_search_complexified(L=L, mu=mu, epsilon=epsilon, n=n,
                                                                                 verbose=self.verbose)
        wc, theory = wc_inexact_gradient_exact_line_search(L=L, mu=mu, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=self.relative_precision * theory)

    def test_inexact_ELS_vs_ELS_modified2(self):
        L, mu, epsilon, n = 2.3, .23, .2, 2

        wc_modified, theory = wc_inexact_gradient_exact_line_search_complexified2(L=L, mu=mu, epsilon=epsilon, n=n,
                                                                                  verbose=self.verbose)
        wc, theory = wc_inexact_gradient_exact_line_search(L=L, mu=mu, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=self.relative_precision * theory)

    def test_inexact_ELS_vs_ELS_modified3(self):
        L, mu, epsilon, n = 2.5, .13, .2, 3

        wc_modified, theory = wc_inexact_gradient_exact_line_search_complexified3(L=L, mu=mu, epsilon=epsilon, n=n,
                                                                                  verbose=self.verbose)
        wc, theory = wc_inexact_gradient_exact_line_search(L=L, mu=mu, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=self.relative_precision * theory)

    def test_randomized_coordinate(self):
        L, d, n = 1, 3, 10
        gamma = 1 / L

        wc_modified, theory = wc_randomized_coordinate_descent_smooth_convex_complexified(L=L, gamma=gamma, d=d, n=n,
                                                                                          verbose=self.verbose)
        wc, theory = wc_randomized_coordinate_descent_smooth_convex(L=L, gamma=gamma, d=d, t=n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=self.relative_precision * theory)

    def test_randomized_coordinate_strongly_convex(self):
        L, mu, d = 1, 0.1, 4
        gamma = 2 / (L + mu)

        wc_modified, theory = wc_randomized_coordinate_descent_smooth_strongly_convex_complexified(L=L, mu=mu,
                                                                                                   gamma=gamma, d=d,
                                                                                                   verbose=self.verbose)

        wc, theory = wc_randomized_coordinate_descent_smooth_strongly_convex(L=L, mu=mu, gamma=gamma, d=d,
                                                                             verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=self.relative_precision * theory)

    def test_gradient_descent_useless_blocks(self):
        L, gamma, n = 1, 1, 5

        wc_modified, theory = wc_gradient_descent_useless_blocks(L=L, gamma=gamma, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, theory, delta=self.relative_precision * theory)

    def test_gradient_descent_blocks_one_block(self):
        L = 1.
        n = 3

        wc_modified, theory = wc_gradient_descent_blocks(L=[L], n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, theory, delta=self.relative_precision * theory)
