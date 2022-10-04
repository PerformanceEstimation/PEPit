import unittest

import PEPit
from PEPit.examples.unconstrained_convex_minimization import wc_proximal_point
from PEPit.examples.composite_convex_minimization import wc_proximal_gradient
from PEPit.examples.unconstrained_convex_minimization import wc_gradient_exact_line_search
from PEPit.examples.unconstrained_convex_minimization import wc_inexact_gradient_exact_line_search

from tests.additional_complexified_examples_tests import wc_proximal_gradient_complexified
from tests.additional_complexified_examples_tests import wc_proximal_point_complexified
from tests.additional_complexified_examples_tests import wc_gradient_exact_line_search_complexified
from tests.additional_complexified_examples_tests import wc_inexact_gradient_exact_line_search_complexified
from tests.additional_complexified_examples_tests import wc_inexact_gradient_exact_line_search_complexified2


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
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_PGD_vs_PGD_modified(self):
        L, mu, gamma, n = 1, 0.1, 1, 2

        wc_modified, theory = wc_proximal_gradient_complexified(L, mu, gamma, n, verbose=self.verbose)
        self.tearDown()
        wc, theory = wc_proximal_gradient(L, mu, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=10 ** -3 * theory)

    def test_PPA_modified(self):
        n, gamma = 2, 1

        wc, theory = wc_proximal_point_complexified(n, gamma, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_PPA_vs_PPA_modified(self):
        n, gamma = 2, 1

        wc_modified, theory = wc_proximal_point_complexified(n, gamma, verbose=self.verbose)
        self.tearDown()
        wc, theory = wc_proximal_point(n, gamma, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=10 ** -3 * theory)

    def test_ELS_modified(self):
        L, mu, n = 3, .3, 3

        wc, theory = wc_gradient_exact_line_search_complexified(L=L, mu=mu, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inexact_ELS_modified(self):
        L, mu, epsilon, n = 2, .05, .2, 2

        wc, theory = wc_inexact_gradient_exact_line_search_complexified(L=L, mu=mu, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inexact_ELS_modified2(self):
        L, mu, epsilon, n = 2, .05, .2, 2

        wc, theory = wc_inexact_gradient_exact_line_search_complexified2(L=L, mu=mu, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_ELS_vs_ELS_modified(self):
        L, mu, n = 1.5, .12, 3

        wc_modified, theory = wc_gradient_exact_line_search_complexified(L=L, mu=mu, n=n, verbose=self.verbose)
        self.tearDown()
        wc, theory = wc_gradient_exact_line_search(L=L, mu=mu, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=10 ** -3 * theory)

    def test_inexact_ELS_vs_ELS_modified(self):
        L, mu, epsilon, n = 2.3, .23, .2, 2

        wc_modified, theory = wc_inexact_gradient_exact_line_search_complexified(L=L, mu=mu, epsilon=epsilon, n=n, verbose=self.verbose)
        self.tearDown()
        wc, theory = wc_inexact_gradient_exact_line_search(L=L, mu=mu, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=10 ** -3 * theory)

    def test_inexact_ELS_vs_ELS_modified2(self):
        L, mu, epsilon, n = 2.3, .23, .2, 2

        wc_modified, theory = wc_inexact_gradient_exact_line_search_complexified2(L=L, mu=mu, epsilon=epsilon, n=n, verbose=self.verbose)
        self.tearDown()
        wc, theory = wc_inexact_gradient_exact_line_search(L=L, mu=mu, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=10 ** -3 * theory)

    def tearDown(self):
        PEPit.reset_classes()
