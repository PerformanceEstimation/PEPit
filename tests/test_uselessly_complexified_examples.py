import unittest

from PEPit.examples.unconstrained_convex_minimization import wc_proximal_point
from PEPit.examples.composite_convex_minimization import wc_proximal_gradient

from tests.additional_complexified_examples_tests import wc_proximal_gradient_complexified
from tests.additional_complexified_examples_tests import wc_proximal_point_complexified


class TestExamples(unittest.TestCase):

    def setUp(self):
        self.n = 6
        self.mu = .1
        self.L = 1

        self.verbose = -1

    def test_PGD_modified(self):
        L, mu, gamma, n = 1, 0.1, 1, 2

        wc, theory = wc_proximal_gradient_complexified(L, mu, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_PGD_vs_PGD_modified(self):
        L, mu, gamma, n = 1, 0.1, 1, 2

        wc_modified, theory = wc_proximal_gradient_complexified(L, mu, gamma, n, verbose=self.verbose)
        wc, theory = wc_proximal_gradient(L, mu, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=10 ** -3 * theory)

    def test_PPA_modified(self):
        n, gamma = 2, 1

        wc, theory = wc_proximal_point_complexified(n, gamma, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_PPA_vs_PPA_modified(self):
        n, gamma = 2, 1

        wc_modified, theory = wc_proximal_point_complexified(n, gamma, verbose=self.verbose)
        wc, theory = wc_proximal_point(n, gamma, verbose=self.verbose)
        self.assertAlmostEqual(wc_modified, wc, delta=10 ** -3 * theory)
