import unittest

import numpy as np

from PEPit.expression import Expression
from PEPit.function import Function
from PEPit.point import Point

import Examples.A_MethodsForUnconstrainedConvexMinimization.ConjugateGradientMethod as CG
import Examples.A_MethodsForUnconstrainedConvexMinimization.FastGradientMethod as inFGM
import Examples.A_MethodsForUnconstrainedConvexMinimization.FastGradientMethod_StronglyConvex as instrFGM
import Examples.A_MethodsForUnconstrainedConvexMinimization.FastProximalPoint as inFPM
import Examples.A_MethodsForUnconstrainedConvexMinimization.GradientExactLineSearch as ELS
import Examples.A_MethodsForUnconstrainedConvexMinimization.HeavyBallMethod as inHBM
import Examples.A_MethodsForUnconstrainedConvexMinimization.InexactAcceleratedGradient as inAGM
import Examples.A_MethodsForUnconstrainedConvexMinimization.InexactGradient as inGD
import Examples.A_MethodsForUnconstrainedConvexMinimization.InexactGradientExactLineSearch as inELS
import Examples.A_MethodsForUnconstrainedConvexMinimization.OptimizedGradientMethod as OGM
import Examples.A_MethodsForUnconstrainedConvexMinimization.RobustMomentumMethod as inRMM
import Examples.A_MethodsForUnconstrainedConvexMinimization.Subgradient as SG
import Examples.A_MethodsForUnconstrainedConvexMinimization.TripleMomentumMethod as inTMM
import Examples.B_MethodsForCompositeConvexMinimization.AcceleratedDouglasRachfordSplitting as inADGS
import Examples.B_MethodsForCompositeConvexMinimization.BregmanProximalPointMethod as inBPPM
import Examples.B_MethodsForCompositeConvexMinimization.ConditionalGradient_FrankWolfe as inCGFW
import Examples.B_MethodsForCompositeConvexMinimization.DouglasRachfordSplitting_1 as inDRS1
import Examples.B_MethodsForCompositeConvexMinimization.DouglasRachfordSplitting_2 as inDRS2
import Examples.B_MethodsForCompositeConvexMinimization.FastProximalGradientMethod as inFPGM
import Examples.B_MethodsForCompositeConvexMinimization.ImprovedInteriorAlgorithm as inIIA
import Examples.B_MethodsForCompositeConvexMinimization.No_Lips_1 as inNL1
import Examples.B_MethodsForCompositeConvexMinimization.No_Lips_2 as inNL2
import Examples.B_MethodsForCompositeConvexMinimization.ThreeOperatorSplitting as inTOS
import Examples.C_MethodsForNonconvexOptimization.GradientMethod as ncGD
import Examples.C_MethodsForNonconvexOptimization.NoLips_1 as ncNL1
import Examples.C_MethodsForNonconvexOptimization.NoLips_2 as ncNL2
import Examples.D_StochasticMethodsForConvexMinimization.SAGA as inSAGA
import Examples.D_StochasticMethodsForConvexMinimization.SGDOverparametrized as inSGD
import Examples.D_StochasticMethodsForConvexMinimization.SGDStronglyConvex as inSGDSC
import Examples.E_MonotoneInclusions.AcceleratedProximalPoint as opAPP
import Examples.E_MonotoneInclusions.DouglasRachfordSplitting as opDRS
import Examples.E_MonotoneInclusions.ProximalPointMethod as opPPM
import Examples.E_MonotoneInclusions.ThreeOperatorSplitting as opTOS
import Examples.F_FixedPointIterations.HalpernIteration as inHI
import Examples.F_FixedPointIterations.KrasnoselskiiMann as inKM
import Examples.G_VerifyPotentialFunctions.FastGradientDescent as potFGD
import Examples.G_VerifyPotentialFunctions.GradientDescent_1 as potGD1
import Examples.G_VerifyPotentialFunctions.GradientDescent_2 as potGD2
import Examples.I_AdaptiveMethods.PolyakSteps_1 as inPS1
import Examples.I_AdaptiveMethods.PolyakSteps_2 as inPS2


class TestExamples(unittest.TestCase):

    def setUp(self):
        self.n = 6
        self.mu = .1
        self.L = 1

        self.verbose = False

    def test_OGM(self):
        L, n = 3, 4

        wc, theory = OGM.wc_ogm(L, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_els(self):
        L, mu, n = 3, .1, 1

        wc, theory = ELS.wc_els(L=L, mu=mu, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_subgd(self):
        M, N = 2, 10
        gamma = 1 / (np.sqrt(N + 1) * M)

        wc, theory = SG.wc_subgd(M=M, N=N, gamma=gamma, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_cg(self):
        L, n = 3, 2

        wc, theory = CG.wc_CG(L=L, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_inexact_els(self):
        L, mu, epsilon, n = 3, .1, .1, 2

        wc, theory = inELS.wc_InexactGrad_ELS(L=L, mu=mu, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_inexact_grad(self):
        L, mu, epsilon, n = 3, .1, .1, 2

        wc, theory = inGD.wc_InexactGrad(L=L, mu=mu, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_inexact_agm(self):
        L, epsilon, n = 3, 0, 5

        wc, theory = inAGM.wc_InexactAGM(L=L, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertAlmostEqual(theory, wc, delta=10 ** -3 * theory)

    def test_inexact_agm_bis(self):
        L, epsilon, n = 2, .1, 5

        wc, theory = inAGM.wc_InexactAGM(L=L, epsilon=epsilon, n=n, verbose=self.verbose)
        self.assertLessEqual(theory, wc * (1 + 10 ** -3))

    def test_hbm(self):
        L, mu, n = 1, .1, 3
        alpha = 1 / (2 * L)  # alpha \in [0, 1/L]
        beta = np.sqrt((1 - alpha * mu) * (1 - L * alpha))

        wc, theory = inHBM.wc_heavyball(mu=mu, L=L, alpha=alpha, beta=beta, n=n, verbose=self.verbose)
        self.assertLessEqual(wc, theory * (1 + 10 ** -3))

    def test_fppa(self):
        A0, n = 1, 3
        gammas = [1, 1, 1]

        wc, theory = inFPM.wc_fppa(A0=A0, gammas=gammas, n=n, verbose=self.verbose)
        self.assertLessEqual(wc, theory * (1 + 10 ** -3))

    def test_tmm(self):
        L, mu, n = 1, .1, 4

        # Compare theoretical rate in epsilon=0 case
        wc, theory = inTMM.wc_tmm(mu=mu, L=L, n=n, verbose=self.verbose)
        self.assertAlmostEqual(theory, wc, delta=10 ** -3 * theory)

    def test_rmm(self):
        L, mu, lam = 1, .1, .5

        # Compare theoretical rate in epsilon=0 case
        wc, theory = inRMM.wc_rmm(mu=mu, L=L, lam=lam, verbose=self.verbose)
        self.assertAlmostEqual(theory, wc, delta=10 ** -3 * theory)

    def test_infgm(self):
        mu, L, n = 0, 1, 10

        wc, theory = inFGM.wc_fgm(mu, L, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_instr_fgm(self):
        L, mu, n = 1, .1, 5

        wc, theory = instrFGM.wc_fgm(mu=mu, L=L, n=n, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_infpgm(self):
        mu, L, n = 0, 1, 5

        wc, theory = inFPGM.wc_fgm(mu, L, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_inadgs(self):
        mu, L, alpha = 0.1, 1, 0.9

        n_list = range(1, 8)
        ref_pesto_bounds = [0.2027, 0.1929, 0.1839, 0.1737, 0.1627, 0.1514, 0.1400, 0.1289]
        for n in n_list:
            wc, _ = inADGS.wc_adrs(mu, L, alpha, n, verbose=self.verbose)
            self.assertAlmostEqual(wc, ref_pesto_bounds[n - 1], delta=10 ** -3 * ref_pesto_bounds[n - 1])

    def test_inbppm(self):
        gamma, n = 3, 5

        wc, theory = inBPPM.wc_bpp(gamma=gamma, n=n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_incgfw(self):
        D, L, n = 1., 1., 10

        wc, theory = inCGFW.wc_cg_fw(L, D, n, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_indrs1(self):
        mu, L, alpha, theta, n = 0.1, 1, 3, 1, 1

        wc, theory = inDRS1.wc_drs(mu, L, alpha, theta, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

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
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_innl2(self):
        L, n = 0.1, 3
        gamma = 1 / L

        wc, theory = inNL2.wc_no_lips2(L, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_intos(self):
        mu, L1, L3, alpha, theta = 0.1, 10, 1, 1, 1
        n_list = range(1, 3)

        ref_pesto_bounds = [0.8304, 0.6895, 0.5726]
        for n in n_list:
            wc, _ = inTOS.wc_tos(mu, L1, L3, alpha, theta, n, verbose=self.verbose)
            self.assertAlmostEqual(wc, ref_pesto_bounds[n - 1], delta=10 ** -3 * ref_pesto_bounds[n - 1])

    def test_ncgd(self):
        L, n = 1, 5
        gamma = 1 / L

        wc, theory = ncGD.wc_gd(L, gamma, n, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_ncnl1(self):
        L, n = 1, 5
        gamma = 1 / L / 2

        wc, theory = ncNL1.wc_no_lips1(L, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_ncnl2(self):
        L, n = 1, 3
        gamma = 1 / L

        wc, theory = ncNL2.wc_no_lips2(L, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_insaga(self):
        L, mu, n = 1, 0.1, 5

        wc, theory = inSAGA.wc_saga(L, mu, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_insgdsc(self):
        L, mu, v, R, n = 1, 0.1, 1, 2, 5
        gamma = 1 / L

        wc, theory = inSGDSC.wc_sgd(L, mu, gamma, v, R, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_insgd(self):
        L, mu, R, n = 1, 0.1, 2, 5
        gamma = 1 / L

        wc, theory = inSGD.wc_sgdo(L, mu, gamma, R, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_opapp(self):
        alpha, n = 2, 10

        wc, theory = opAPP.wc_ppm(alpha, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_opppm(self):
        alpha, n = 2, 3

        wc, theory = opPPM.wc_ppm(alpha, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_opdrs(self):
        L, mu, alpha, theta = 1, 0.1, 1.3, 0.9

        wc, theory = opDRS.wc_drs(L, mu, alpha, theta, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_optos(self):
        L, mu, beta, alpha, theta = 1, 0.1, 1, 1.3, 0.9
        n_list = range(1, 1)

        ref_pesto_bounds = [0.7797]
        for n in n_list:
            wc, _ = opTOS.wc_tos(L, mu, beta, alpha, theta, verbose=self.verbose)
            self.assertAlmostEqual(wc, ref_pesto_bounds[n - 1], delta=10 ** -3 * ref_pesto_bounds[n - 1])

    def test_inhi(self):
        n = 10

        wc, theory = inHI.wc_halpern(n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_inkm(self):
        n = 10

        wc, theory = inKM.wc_km(n, verbose=self.verbose)
        self.assertLessEqual(wc, theory)

    def test_potgd1(self):
        L, n = 1, 10
        gamma = 1 / L

        wc, theory = potGD1.wc_gd_lyapunov_1(L, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 + theory)

    def test_potgd2(self):
        L, n = 1, 10
        gamma = 1 / L

        wc, theory = potGD2.wc_gd_lyapunov_2(L, gamma, n, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 + theory)

    def test_potfgd(self):
        L, lam = 1, 10
        gamma = 1 / L

        wc, theory = potFGD.wc_gd_lyapunov(L, gamma, lam, verbose=self.verbose)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 + theory)

    def test_inps1(self):
        L, mu = 1, 0.1
        gamma = 2 / L

        wc, theory = inPS1.wc_ps_1(L, mu, gamma, verbose=self.verbose)
        self.assertLessEqual(wc, theory + 10 ** -3)

    def test_inps2(self):
        L, mu = 1, 0.1
        gamma = 2 / L

        wc, theory = inPS2.wc_ps_2(L, mu, gamma, verbose=self.verbose)
        self.assertLessEqual(wc, theory + 10 ** -3)

    def tearDown(self):
        Point.counter = 0
        Expression.counter = 0
        Function.counter = 0

if __name__ == '__main__':
    unittest.main()
