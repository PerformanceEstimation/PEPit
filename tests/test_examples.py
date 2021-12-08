import unittest

import numpy as np

from PEPit.expression import Expression
from PEPit.function import Function
from PEPit.point import Point

import PEPit.examples.a_methods_for_unconstrained_convex_minimization.conjugate_gradient_method as CG
import PEPit.examples.a_methods_for_unconstrained_convex_minimization.gradient_descent as inGD
import PEPit.examples.a_methods_for_unconstrained_convex_minimization.fast_gradient_method as inFGM
import PEPit.examples.a_methods_for_unconstrained_convex_minimization.fast_gradient_method_strongly_convex as instrFGM
import PEPit.examples.a_methods_for_unconstrained_convex_minimization.fast_proximal_point as inFPM
import PEPit.examples.a_methods_for_unconstrained_convex_minimization.proximal_point_method as inPPM
import PEPit.examples.a_methods_for_unconstrained_convex_minimization.gradient_exact_line_search as ELS
import PEPit.examples.a_methods_for_unconstrained_convex_minimization.heavy_ball_method as inHBM
import PEPit.examples.a_methods_for_unconstrained_convex_minimization.inexact_accelerated_gradient as inAGM
import PEPit.examples.a_methods_for_unconstrained_convex_minimization.inexact_gradient_descent as inEGD
import PEPit.examples.a_methods_for_unconstrained_convex_minimization.inexact_gradient_exact_line_search as inELS
import PEPit.examples.a_methods_for_unconstrained_convex_minimization.optimized_gradient_method as OGM
import PEPit.examples.a_methods_for_unconstrained_convex_minimization.robust_momentum_method as inRMM
import PEPit.examples.a_methods_for_unconstrained_convex_minimization.subgradient_method as SG
import PEPit.examples.a_methods_for_unconstrained_convex_minimization.triple_momentum_method as inTMM
import PEPit.examples.b_methods_for_composite_convex_minimization.accelerated_douglas_rachford_splitting as inADGS
import PEPit.examples.b_methods_for_composite_convex_minimization.bregman_proximal_point_method as inBPPM
import PEPit.examples.b_methods_for_composite_convex_minimization.conditional_gradient_frank_wolfe as inCGFW
import PEPit.examples.b_methods_for_composite_convex_minimization.douglas_rachford_splitting_1 as inDRS1
import PEPit.examples.b_methods_for_composite_convex_minimization.douglas_rachford_splitting_2 as inDRS2
import PEPit.examples.b_methods_for_composite_convex_minimization.fast_proximal_gradient_method as inFPGM
import PEPit.examples.b_methods_for_composite_convex_minimization.improved_interior_algorithm as inIIA
import PEPit.examples.b_methods_for_composite_convex_minimization.no_lips_1 as inNL1
import PEPit.examples.b_methods_for_composite_convex_minimization.no_lips_2 as inNL2
import PEPit.examples.b_methods_for_composite_convex_minimization.three_operator_splitting as inTOS
import PEPit.examples.c_methods_for_nonconvex_optimization.gradient_method as ncGD
import PEPit.examples.c_methods_for_nonconvex_optimization.no_lips_1 as ncNL1
import PEPit.examples.c_methods_for_nonconvex_optimization.no_lips_2 as ncNL2
import PEPit.examples.d_stochastic_methods_for_convex_minimization.SAGA as inSAGA
import PEPit.examples.d_stochastic_methods_for_convex_minimization.SGD_overparametrized as inSGD
import PEPit.examples.d_stochastic_methods_for_convex_minimization.SGD_strongly_convex as inSGDSC
import PEPit.examples.d_stochastic_methods_for_convex_minimization.point_SAGA as inPSAGA
import PEPit.examples.e_monotone_inclusions.accelerated_proximal_point as opAPP
import PEPit.examples.e_monotone_inclusions.douglas_rachford_splitting as opDRS
import PEPit.examples.e_monotone_inclusions.proximal_point_method as opPPM
import PEPit.examples.e_monotone_inclusions.three_operator_splitting as opTOS
import PEPit.examples.f_fixed_point_iterations.halpern_iteration as inHI
import PEPit.examples.f_fixed_point_iterations.krasnoselskii_mann as inKM
import PEPit.examples.g_verify_potential_functions.fast_gradient_descent as potFGD
import PEPit.examples.g_verify_potential_functions.gradient_descent_1 as potGD1
import PEPit.examples.g_verify_potential_functions.gradient_descent_2 as potGD2
import PEPit.examples.i_adaptive_methods.polyak_steps_1 as inPS1
import PEPit.examples.i_adaptive_methods.polyak_steps_2 as inPS2
import PEPit.examples.j_low_dimensional_worst_cases_scenarios.inexact_gradient as inLDIGD
import PEPit.examples.j_low_dimensional_worst_cases_scenarios.optimized_gradient_method as inLDOGM
import PEPit.examples.h_inexact_proximal_methods.accelerated_hybrid_proximal_extra_gradient as inAHPE
import PEPit.examples.h_inexact_proximal_methods.accelerated_inexact_forward_backward as inAIFB
import PEPit.examples.h_inexact_proximal_methods.optimized_relatively_inexact_proximal_point_algorithm as inORIPPA
import PEPit.examples.h_inexact_proximal_methods.partially_inexact_douglas_rachford_splitting as inPIDRS
import PEPit.examples.h_inexact_proximal_methods.relatively_inexact_proximal_point_algorithm as inRIPP


class TestExamples(unittest.TestCase):

    def setUp(self):
        self.n = 6
        self.mu = .1
        self.L = 1
        self.verbose = False
        self.relative_precision = 10 ** -3
        self.absolute_precision = 5 * 10 ** -5

    def test_OGM(self):
        L, n = 3, 4

        wc, theory = OGM.wc_ogm(L, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_GD(self):
        L, n = 3, 4
        gamma = 1/L
        
        wc, theory = inGD.wc_gd(L, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)
        
    def test_els(self):
        L, mu, n = 3, .1, 1

        wc, theory = ELS.wc_els(L=L, mu=mu, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_subgd(self):
        M, N = 2, 10
        gamma = 1 / (np.sqrt(N + 1) * M)

        wc, theory = SG.wc_subgd(M=M, N=N, gamma=gamma, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_cg(self):
        L, n = 3, 2

        wc, theory = CG.wc_CG(L=L, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inexact_els(self):
        L, mu, epsilon, n = 3, .1, .1, 2

        wc, theory = inELS.wc_InexactGrad_ELS(L=L, mu=mu, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inexact_grad(self):
        L, mu, epsilon, n = 3, .1, .1, 2

        wc, theory = inEGD.wc_inexact_gradient_descent(L=L, mu=mu, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)
        
    def test_proximalpointmethod(self):
        n, gamma = 3, .1

        wc, theory = inPPM.wc_ppa(gamma=gamma, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_lowdimensional_OGM(self):
        L, n = 3, 4

        wc, theory = inLDOGM.wc_ogm(L, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_lowdimensional_inexact_grad(self):
        L, mu, epsilon, n = 3, .1, .1, 2

        wc, theory = inLDIGD.wc_InexactGrad(L=L, mu=mu, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inexact_agm(self):
        L, epsilon, n = 3, 0, 5

        wc, theory = inAGM.wc_InexactAGM(L=L, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertAlmostEqual(theory, wc, delta=self.relative_precision + theory)

    def test_inexact_agm_bis(self):
        L, epsilon, n = 2, .1, 5

        wc, theory = inAGM.wc_InexactAGM(L=L, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertLessEqual(theory, wc * (1 + self.relative_precision))

    def test_hbm(self):
        L, mu, n = 1, .1, 3
        alpha = 1 / (2 * L)  # alpha \in [0, 1/L]
        beta = np.sqrt((1 - alpha * mu) * (1 - L * alpha))

        wc, theory = inHBM.wc_heavyball(mu=mu, L=L, alpha=alpha, beta=beta, n=n, verbose=self.verbose)
        self.assertLessEqual(wc, theory * (1 + self.relative_precision))

    def test_fppa(self):
        A0, n = 1, 3
        gammas = [1, 1, 1]

        wc, theory = inFPM.wc_fppa(A0=A0, gammas=gammas, n=n, verbose=self.verbose)
        self.assertLessEqual(wc, theory * (1 + self.relative_precision))

    def test_tmm(self):
        L, mu, n = 1, .1, 4

        # Compare theoretical rate in epsilon=0 case
        wc, theory = inTMM.wc_tmm(mu=mu, L=L, n=n, verbose=self.verbose)
        self.assertAlmostEqual(theory, wc, delta=self.relative_precision * theory)

    def test_rmm(self):
        L, mu, lam = 1, .1, .5

        # Compare theoretical rate in epsilon=0 case
        wc, theory = inRMM.wc_rmm(mu=mu, L=L, lam=lam, verbose=self.verbose)
        self.assertAlmostEqual(theory, wc, delta=self.relative_precision * theory)

    def test_infgm(self):
        mu, L, n = 0, 1, 10

        wc, theory = inFGM.wc_fgm(mu, L, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_instr_fgm(self):
        L, mu, n = 1, .1, 5

        wc, theory = instrFGM.wc_fgm(mu=mu, L=L, n=n, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_infpgm(self):
        mu, L, n = 0, 1, 5

        wc, theory = inFPGM.wc_fgm(mu, L, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inadgs(self):
        mu, L, alpha = 0.1, 1, 0.9

        n_list = range(1, 8)
        ref_pesto_bounds = [0.2027, 0.1929, 0.1839, 0.1737, 0.1627, 0.1514, 0.1400, 0.1289]
        for n in n_list:
            wc, _ = inADGS.wc_adrs(mu, L, alpha, n, verbose=self.verbose)
            self.assertAlmostEqual(wc, ref_pesto_bounds[n - 1], delta=self.relative_precision * ref_pesto_bounds[n - 1])

    def test_inbppm(self):
        gamma, n = 3, 5

        wc, theory = inBPPM.wc_bpp(gamma=gamma, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_incgfw(self):
        D, L, n = 1., 1., 10

        wc, theory = inCGFW.wc_cg_fw(L, D, n, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_indrs1(self):
        mu, L, alpha, theta, n = 0.1, 1, 3, 1, 1

        wc, theory = inDRS1.wc_drs(mu, L, alpha, theta, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_indrs2(self):
        L, alpha, theta, n = 1, 1, 1, 10

        wc, theory = inDRS2.wc_drs_2(L, alpha, theta, n, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_iniia(self):
        L, mu, c, n = 1, 1, 1, 5
        lam = 1 / L

        wc, theory = inIIA.wc_iipp(L, mu, c, lam, n, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_innl1(self):
        L, n = 1, 3
        gamma = 1 / L / 2

        wc, theory = inNL1.wc_no_lips1(L, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_innl2(self):
        L, n = 0.1, 3
        gamma = 1 / L

        wc, theory = inNL2.wc_no_lips2(L, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_intos(self):
        mu, L1, L3, alpha, theta = 0.1, 10, 1, 1, 1
        n_list = range(1, 3)

        ref_pesto_bounds = [0.8304, 0.6895, 0.5726]
        for n in n_list:
            wc, _ = inTOS.wc_tos(mu, L1, L3, alpha, theta, n, verbose=self.verbose)
            self.assertAlmostEqual(wc, ref_pesto_bounds[n - 1], delta=self.relative_precision * ref_pesto_bounds[n - 1])

    def test_ncgd(self):
        L, n = 1, 5
        gamma = 1 / L

        wc, theory = ncGD.wc_gd(L, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_ncnl1(self):
        L, n = 1, 5
        gamma = 1 / L / 2

        wc, theory = ncNL1.wc_no_lips1(L, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_ncnl2(self):
        L, n = 1, 3
        gamma = 1 / L

        wc, theory = ncNL2.wc_no_lips2(L, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_insaga(self):
        L, mu, n = 1, 0.1, 5

        wc, theory = inSAGA.wc_saga(L, mu, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_insgdsc(self):
        L, mu, v, R, n = 1, 0.1, 1, 2, 5
        gamma = 1 / L

        wc, theory = inSGDSC.wc_sgd(L, mu, gamma, v, R, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_insgd(self):
        L, mu, n = 1, 0.1, 5
        gamma = 1 / L

        wc, theory = inSGD.wc_sgdo(L, mu, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inpsaga(self):
        L, mu, n = 1, 0.1, 10

        wc, theory = inPSAGA.wc_psaga(L, mu, n, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_opapp(self):
        alpha, n = 2, 10

        wc, theory = opAPP.wc_ppm(alpha, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_opppm(self):
        alpha, n = 2, 3

        wc, theory = opPPM.wc_ppm(alpha, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_opdrs(self):
        L, mu, alpha, theta = 1, 0.1, 1.3, 0.9

        wc, theory = opDRS.wc_drs(L, mu, alpha, theta, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_optos(self):
        L, mu, beta, alpha, theta = 1, 0.1, 1, 1.3, 0.9
        n_list = range(1, 1)

        ref_pesto_bounds = [0.7797]
        for n in n_list:
            wc, _ = opTOS.wc_tos(L, mu, beta, alpha, theta, verbose=self.verbose)
            self.assertAlmostEqual(wc, ref_pesto_bounds[n - 1], delta=self.relative_precision * ref_pesto_bounds[n - 1])

    def test_inhi(self):
        n = 10

        wc, theory = inHI.wc_halpern(n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inkm(self):
        n = 10

        wc, theory = inKM.wc_km(n, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_potgd1(self):
        L, n = 1, 10
        gamma = 1 / L

        wc, theory = potGD1.wc_gd_lyapunov_1(L, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_potgd2(self):
        L, n = 1, 10
        gamma = 1 / L

        wc, theory = potGD2.wc_gd_lyapunov_2(L, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_potfgd(self):
        L, lam = 1, 10
        gamma = 1 / L

        wc, theory = potFGD.wc_gd_lyapunov(L, gamma, lam, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_inps1(self):
        L, mu = 1, 0.1
        gamma = 2 / L

        wc, theory = inPS1.wc_ps_1(L, mu, gamma, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_inps2(self):
        L, mu = 1, 0.1
        gamma = 2 / L

        wc, theory = inPS2.wc_ps_2(L, mu, gamma, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_inAHPE(self):
        mu, gamma, sigma, A0 = 1, 1, 1, 10

        wc, theory = inAHPE.wc_ahpe(mu, gamma, sigma, A0, verbose=self.verbose)
        self.assertLessEqual(wc, self.absolute_precision)

    def test_inAIFB(self):
        mu, L, sigma, zeta, xi, A0 = 1, 2, 0.2, 0.9, 3, 1
        gamma = (1 - sigma ** 2) / L

        wc, theory = inAIFB.wc_aifb(mu, L, gamma, sigma, xi, zeta, A0, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.absolute_precision)

    def test_inORIPPA(self):
        gamma, sigma, n = 2, 3, 10

        wc, theory = inORIPPA.wc_orippm(n, gamma, sigma, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inPIDRS(self):
        mu, L, gamma, sigma, n = 1, 5., 1.4, 0.2, 5

        wc, theory = inPIDRS.wc_pidrs(mu, L, n, gamma, sigma, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=self.relative_precision * theory)

    def test_inRIPP(self):
        gamma, sigma, n = 2, 0.3, 5

        wc, theory = inRIPP.wc_rippm(n, gamma, sigma, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def tearDown(self):
        Point.counter = 0
        Expression.counter = 0
        Function.counter = 0


if __name__ == '__name__':
    unittest.main()
