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

    def test_OGM(self):
        L, n = 3, 4
        wc, theory = OGM.wc_ogm(L, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_els(self):

        L, mu, n = 3, .1, 1

        wc, theory = ELS.wc_els(L=L, mu=mu, n=n)

        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_subgd(self):

        M, N = 2, 10
        gamma = 1 / (np.sqrt(N + 1) * M)

        wc, theory = SG.wc_subgd(M=M, N=N, gamma=gamma)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_cg(self):

        L, n = 3, 2

        wc, theory = CG.wc_CG(L=L, n=n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_inexact_els(self):

        L, mu, epsilon, n = 3, .1, .1, 2

        wc, theory = inELS.wc_InexactGrad_ELS(L=L, mu=mu, epsilon=epsilon, n=n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_inexact_grad(self):

        L, mu, epsilon, n = 3, .1, .1, 2

        wc, theory = inGD.wc_InexactGrad(L=L, mu=mu, epsilon=epsilon, n=n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_inexact_agm(self):

        L, epsilon, n = 3, 0, 5
        wc, theory = inAGM.wc_InexactAGM(L=L, epsilon=epsilon, n=n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_inexact_agm_bis(self):

        L, epsilon, n = 2, .1, 5

        # Must run
        _, _ = inAGM.wc_InexactAGM(L=L, epsilon=epsilon, n=n)

        # Compare theoretical rate in epsilon=0 case
        wc, theory = inAGM.wc_InexactAGM(L=L, epsilon=0, n=n)
        self.assertLessEqual(wc, theory+10**-3)

    # def test_(self):
    #
    #     mu = 0.1
    #     L = 1.
    #     alpha = 1 / (2 * L)  # alpha \in [0, 1/L]
    #     beta = np.sqrt((1 - alpha * mu) * (1 - L * alpha))
    #     n = 3
    #     wc = inHBM.wc_heavyball(mu=mu, L=L, alpha=alpha, beta=beta, n=n)
    #
    # def test_(self):
    #
    #     A0, n = 1, 3
    #     gammas = [1, 1, 1]
    #
    #     wc = inFPM.wc_fppa(A0=A0, gammas=gammas, n=n)
    #
    # def test_(self):
    #
    #     L, mu, n = 1, .1, 1
    #
    #     wc = inTMM.wc_tmm(mu=mu, L=L, n=n)
    #
    # def test_(self):
    #
    #     L, mu, n = 1, .1, 1
    #     lam = .5
    #
    #     wc = inRMM.wc_rmm(mu=mu, L=L, lam=lam)

    def test_infgm(self):

        mu, L, n = 0, 1, 10

        wc, theory = inFGM.wc_fgm(mu, L, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_instr_fgm(self):

        L, mu, n = 1, .1, 5

        wc, theory = instrFGM.wc_fgm(mu=mu, L=L, n=n)
        self.assertLessEqual(wc, theory)

    def test_infpgm(self):

        mu, L, n = 0, 1, 5

        wc, theory = inFPGM.wc_fgm(mu, L, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_inadgs(self):

        mu, L, n, alpha = 0.1, 1, 2, 0.9

        wc, theory = inADGS.wc_adrs(mu, L, alpha, n)
        self.assertLessEqual(wc, theory)

    def test_inbppm(self):

        gamma, n = 3, 5

        wc, theory = inBPPM.wc_bpp(gamma=gamma, n=n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_incgfw(self):

        D, L, n = 1., 1., 10

        wc, theory = inCGFW.wc_cg_fw(L, D, n)
        self.assertLessEqual(wc, theory)

    def test_indrs1(self):

        mu, L, alpha, theta, n = 0.1, 1, 3, 2, 1

        wc, theory = inDRS1.wc_drs(mu, L, alpha, theta, n)
        self.assertLessEqual(wc, theory)

    def test_indrs2(self):

        L, alpha, theta, n = 1, 1, 1, 10

        wc, theory = inDRS2.wc_drs_2(L, alpha, theta, n)
        self.assertLessEqual(wc, theory)

    def test_iniia(self):

        L, mu, c, n = 1, 1, 1, 5
        lam = 1 / L

        wc, theory = inIIA.wc_iipp(L, mu, c, lam, n)
        self.assertLessEqual(wc, theory)

    def test_innl1(self):

        L, n = 1, 3
        gamma = 1 / L / 2

        wc, theory = inNL1.wc_no_lips1(L, gamma, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_innl2(self):

        L, n = 0.1, 3
        gamma = 1 / L

        wc, theory = inNL2.wc_no_lips2(L, gamma, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_intos(self):

        mu, L1, L3, alpha, theta, n = 0.1, 10, 1, 1, 1, 4

        wc, theory = inTOS.wc_tos(mu, L1, L3, alpha, theta, n)
        self.assertLessEqual(wc, theory)

    def test_ncgd(self):

        L, n = 1, 5
        gamma = 1 / L

        wc, theory = ncGD.wc_gd(L, gamma, n)
        self.assertLessEqual(wc, theory)

    def test_ncnl1(self):

        L, n = 1, 5
        gamma = 1 / L / 2

        wc, theory = ncNL1.wc_no_lips1(L, gamma, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_ncnl2(self):

        L, n = 1, 3
        gamma = 1 / L

        wc, theory = ncNL2.wc_no_lips2(L, gamma, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_insaga(self):

        L, mu, n = 1, 0.1, 5

        wc, theory = inSAGA.wc_saga(L, mu, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_insgdsc(self):

        L, mu, v, R, n = 1, 0.1, 1, 2, 5
        gamma = 1 / L

        wc, theory = inSGDSC.wc_sgd(L, mu, gamma, v, R, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_insgd(self):

        L, mu, R, n = 1, 0.1, 2, 5
        gamma = 1 / L

        wc, theory = inSGD.wc_sgdo(L, mu, gamma, R, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_opapp(self):

        alpha, n = 2, 10

        wc, theory = opAPP.wc_ppm(alpha, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_opppm(self):

        alpha, n = 2, 3

        wc, theory = opPPM.wc_ppm(alpha, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_opdrs(self):

        L, mu, alpha, theta = 1, 0.1, 1.3, 0.9

        wc, theory = opDRS.wc_drs(L, mu, alpha, theta)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    # def test_optos(self):
    #
    #     L, mu, beta, alpha, theta = 1, 0.1, 1, 1.3, 0.9
    #
    #     wc, theory = opTOS.wc_tos(L, mu, beta, alpha, theta)

    def test_inhi(self):

        L, n = 1, 10

        wc, theory = inHI.wc_halpern(L, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_inkm(self):

        L, n = 1, 10

        wc, theory = inKM.wc_km(L, n)
        self.assertLessEqual(wc, theory)

    def test_potgd1(self):

        L, n = 1, 10
        gamma = 1 / L

        wc, theory = potGD1.wc_gd_lyapunov_1(L, gamma, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_potgd2(self):

        L, n = 1, 10
        gamma = 1 / L

        wc, theory = potGD2.wc_gd_lyapunov_2(L, gamma, n)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_potfgd(self):

        L, lam = 1, 10
        gamma = 1 / L

        wc, theory = potFGD.wc_gd_lyapunov(L, gamma, lam)
        self.assertAlmostEqual(wc, theory, delta=10 ** -3 * theory)

    def test_inps1(self):

        L, mu = 1, 0.1
        gamma = 1 / L

        wc, theory = inPS1.wc_ps_1(L, mu, gamma)
        self.assertLessEqual(wc, theory+10**-3)

    def test_inps2(self):

        L, mu = 1, 0.1
        gamma = 1 / L

        wc, theory = inPS2.wc_ps_2(L, mu, gamma)
        self.assertLessEqual(wc, theory+10**-3)

    def tearDown(self):
        Point.counter = 0
        Expression.counter = 0
        Function.counter = 0
