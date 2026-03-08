import unittest

import numpy as np
from PEPit import PEP
from PEPit.point import Point
from PEPit.functions.convex_function import ConvexFunction
from PEPit.functions.convex_function import SmoothConvexFunction
from PEPit.functions.convex_function import SmoothStronglyConvexFunction
from PEPit.functions.convex_function import StronglyConvexFunction
from PEPit.functions.convex_function import SmoothFunction


class TestInterpolator(unittest.TestCase):

    def setUp(self):
    
        self.func1 = ConvexFunction()
        self.func2 = SmoothConvexFunction(L=1.)
        self.func3 = SmoothStronglyConvexFunction(L=1.2,mu=.1)
        self.func4 = StronglyConvexFunction(mu=.25)
        self.func5 = SmoothFunction(L=1.5)
        
        self.interp1 = func1.get_interpolator(options='lowest')
        pt1 = Point()
        self.interp2 = func2.get_interpolator(options='highest')
        self.interp3 = func3.get_interpolator()
        self.interp4 = func4.get_interpolator()
        pt2 = Point()
        pt3 = Point()
        self.interp5 = func5.get_interpolator()
        

    def test_is_instance(self):

        self.assertIsInstance(self.interp1, Interpolator)
        self.assertIsInstance(self.interp2, Interpolator)
        self.assertIsInstance(self.interp3, Interpolator)
        self.assertIsInstance(self.interp4, Interpolator)
        self.assertIsInstance(self.interp5, Interpolator)

        self.assertIsInstance(self.interp1, SmoothStronglyConvexInterpolator)
        self.assertIsInstance(self.interp2, SmoothStronglyConvexInterpolator)
        self.assertIsInstance(self.interp3, SmoothStronglyConvexInterpolator)
        self.assertIsInstance(self.interp4, SmoothStronglyConvexInterpolator)
        self.assertIsInstance(self.interp5, SmoothStronglyConvexInterpolator)

    def test_values(self):

        self.assertEqual(self.interp1.L, np.inf)
        self.assertEqual(self.interp2.L, 1.)
        self.assertEqual(self.interp3.L, 1.2)
        self.assertEqual(self.interp4.L, np.inf)
        self.assertEqual(self.interp5.L, 1.5)

        self.assertEqual(self.interp1.mu, 0)
        self.assertEqual(self.interp2.mu, 0)
        self.assertEqual(self.interp3.mu, .1)
        self.assertEqual(self.interp4.mu, .25)
        self.assertEqual(self.interp5.mu, -self.interp5.L)
        
    def test_references(self):

        self.assertEqual(self.interp1.func, func1)
        self.assertEqual(self.interp2.func, func2)
        self.assertEqual(self.interp3.func, func3)
        self.assertEqual(self.interp4.func, func4)
        self.assertEqual(self.interp5.func, func5)
    

    def test_dimensions(self):

        self.assertEqual(self.interp1.d, 0)
        self.assertEqual(self.interp2.d, 1)
        self.assertEqual(self.interp3.d, 1)
        self.assertEqual(self.interp4.d, 1)
        self.assertEqual(self.interp5.d, 3)

    def test_options(self):

        self.assertEqual(self.interp1.options, 'lowest')
        self.assertEqual(self.interp2.options, 'highest')

    def test_naive_value1(self):
    
        # On the problem below, the only possible interpolable point will 
        problem = PEP()
        # Declare a smooth strongly convex function
        f = problem.declare_function(SmoothConvexFunction, L=L,mu=0)
        xs = f.stationary_point
        x0 = problem.set_initial_point()
        f0, fs = f(x0), f(xs)
        problem.set_initial_condition((x0-xs)**2 <= 1)
        problem.set_performance_metric(f0-fs)
        pepit_tau = problem.solve(verbose=0)
        
        x0_val = x0.eval()
        xs_val = xs.eval()
        x_mid = 1/2 * (x0_val+xs_val)
        
        f_interp_low = f.get_interpolator(options='lowest')
        f_mid_low = f_interp_low(x_mid)
        f_interp_high = f.get_interpolator(options='highest')
        f_mid_high = f_interp_high(x_mid)
        
        self.assertAlmostEqual(f_mid_low, f_mid_high, delta=1e-5)
