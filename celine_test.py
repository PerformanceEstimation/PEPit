import Examples.A_MethodsForUnconstrainedConvexMinimization.OptimizedGradientMethod as OGM
import Examples.A_MethodsForUnconstrainedConvexMinimization.GradientExactLineSearch as ELS
import Examples.A_MethodsForUnconstrainedConvexMinimization.Subgradient as SG
import Examples.A_MethodsForUnconstrainedConvexMinimization.ConjugateGradientMethod as CG
import Examples.A_MethodsForUnconstrainedConvexMinimization.InexactGradientExactLineSearch as inELS
import Examples.A_MethodsForUnconstrainedConvexMinimization.InexactGradient as inGD
import Examples.A_MethodsForUnconstrainedConvexMinimization.InexactAcceleratedGradient as inAGM
import Examples.A_MethodsForUnconstrainedConvexMinimization.HeavyBallMethod as inHBM
import Examples.A_MethodsForUnconstrainedConvexMinimization.FastProximalPoint as inFPM
import Examples.A_MethodsForUnconstrainedConvexMinimization.TripleMomentumMethod as inTMM
import Examples.A_MethodsForUnconstrainedConvexMinimization.RobustMomentumMethod as inRMM
import Examples.A_MethodsForUnconstrainedConvexMinimization.FastGradientMethod as inFGM
import Examples.A_MethodsForUnconstrainedConvexMinimization.FastGradientMethod_StronglyConvex as instrFGM
import Examples.B_MethodsForCompositeConvexMinimization.FastProximalGradientMethod as inFPGM
import Examples.B_MethodsForCompositeConvexMinimization.AcceleratedDouglasRachfordSplitting as inADGS
import Examples.B_MethodsForCompositeConvexMinimization.BregmanProximalPointMethod as inBPPM
import Examples.B_MethodsForCompositeConvexMinimization.ConditionalGradient_FrankWolfe as inCGFW
import Examples.B_MethodsForCompositeConvexMinimization.DouglasRachfordSplitting_1 as inDRS1
import Examples.B_MethodsForCompositeConvexMinimization.DouglasRachfordSplitting_2 as inDRS2
import Examples.B_MethodsForCompositeConvexMinimization.ImprovedInteriorAlgorithm as inIIA
import Examples.B_MethodsForCompositeConvexMinimization.No_Lips_1 as inNL1
import Examples.B_MethodsForCompositeConvexMinimization.No_Lips_2 as inNL2
import Examples.B_MethodsForCompositeConvexMinimization.ThreeOperatorSplitting as inTOS
import Examples.C_MethodsForNonconvexOptimization.GradientMethod as ncGD
import Examples.C_MethodsForNonconvexOptimization.NoLips_1 as ncNL1
import Examples.C_MethodsForNonconvexOptimization.NoLips_2 as ncNL2
import Examples.D_StochasticMethodsForConvexMinimization.SAGA as inSAGA
import Examples.D_StochasticMethodsForConvexMinimization.SGDStronglyConvex as inSGDSC
import Examples.D_StochasticMethodsForConvexMinimization.SGDOverparametrized as inSGD
import Examples.E_MonotoneInclusions.ThreeOperatorSplitting as opTOS
import Examples.E_MonotoneInclusions.ProximalPointMethod as opPPM
import Examples.E_MonotoneInclusions.AcceleratedProximalPoint as opAPP
import Examples.E_MonotoneInclusions.DouglasRachfordSplitting as opDRS
import Examples.F_FixedPointIterations.HalpernIteration as inHI
import Examples.F_FixedPointIterations.KrasnoselskiiMann as inKM
import Examples.G_VerifyPotentialFunctions.GradientDescent_1 as potGD1
import Examples.G_VerifyPotentialFunctions.GradientDescent_2 as potGD2
import Examples.G_VerifyPotentialFunctions.FastGradientDescent as potFGD
import Examples.I_AdaptiveMethods.PolyakSteps_1 as inPS1
import Examples.I_AdaptiveMethods.PolyakSteps_2 as inPS2

import numpy as np

import time

L, n = 3, 4
start = time.time()
wc, theory = OGM.wc_ogm(L, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu, n = 3, .1, 1

start = time.time()
wc, theory = ELS.wc_els(L=L, mu=mu, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

M, N = 2, 10
gamma = 1 / (np.sqrt(N + 1) * M)

start = time.time()
wc, theory = SG.wc_subgd(M=M, N=N, gamma=gamma)
end = time.time()
print('Timing:', end - start, '[s]')

L, n = 3, 2

start = time.time()
wc, theory = CG.wc_CG(L=L, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu, epsilon, n = 3, .1, .1, 2

start = time.time()
wc, theory = inELS.wc_InexactGrad_ELS(L=L, mu=mu, epsilon=epsilon, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu, epsilon, n = 3, .1, .1, 2

start = time.time()
wc, theory = inGD.wc_InexactGrad(L=L, mu=mu, epsilon=epsilon, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

L, epsilon, n = 3, 0, 5

start = time.time()
wc, theory = inAGM.wc_InexactAGM(L=L, epsilon=epsilon, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

L, epsilon, n = 2, .1, 5

start = time.time()
wc, theory = inAGM.wc_InexactAGM(L=L, epsilon=epsilon, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

mu = 0.1
L = 1.
alpha = 1 / (2 * L)  # alpha \in [0, 1/L]
beta = np.sqrt((1 - alpha * mu) * (1 - L * alpha))
n = 3

start = time.time()

wc = inHBM.wc_heavyball(mu=mu, L=L, alpha=alpha, beta=beta, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

A0, n = 1, 3
gammas = [1, 1, 1]

start = time.time()

wc = inFPM.wc_fppa(A0=A0, gammas=gammas, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu, n = 1, .1, 1

start = time.time()

wc = inTMM.wc_tmm(mu=mu, L=L, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu, n = 1, .1, 1
lam = .5

start = time.time()

wc = inRMM.wc_rmm(mu=mu, L=L, lam=lam)
end = time.time()
print('Timing:', end - start, '[s]')

mu, L, n = 0, 1, 10
start = time.time()
wc, theory = inFGM.wc_fgm(mu, L, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu, n = 1, .1, 5

start = time.time()
wc, theory = instrFGM.wc_fgm(mu=mu, L=L, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

mu, L, n = 0, 1, 5
start = time.time()
wc, theory = inFPGM.wc_fgm(mu, L, n)
end = time.time()
print('Timing:', end - start, '[s]')

mu, L, n, alpha = 0.1, 1, 2, 0.9
start = time.time()
wc, theory = inADGS.wc_adrs(mu, L, alpha, n)
end = time.time()
print('Timing:', end - start, '[s]')

gamma, n = 3, 5
start = time.time()
wc, theory = inBPPM.wc_bpp(gamma=gamma, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

D, L, n = 1., 1., 10
start = time.time()
wc, theory = inCGFW.wc_cg_fw(L, D, n)
end = time.time()
print('Timing:', end - start, '[s]')

mu, L, alpha, theta, n = 0.1, 1, 3, 2, 1
start = time.time()
wc, theory = inDRS1.wc_drs(mu, L, alpha, theta, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, alpha, theta, n = 1, 1, 1, 10
start = time.time()
wc, theory = inDRS2.wc_drs_2(L, alpha, theta, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu, c, n = 1, 1, 1, 5
lam = 1 / L
start = time.time()
wc, theory = inIIA.wc_iipp(L, mu, c, lam, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, n = 1, 3
gamma = 1 / L / 2
start = time.time()
wc, theory = inNL1.wc_no_lips1(L, gamma, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, n = 0.1, 3
gamma = 1 / L
start = time.time()
wc, theory = inNL2.wc_no_lips2(L, gamma, n)
end = time.time()
print('Timing:', end - start, '[s]')

mu, L1, L3, alpha, theta, n = 0.1, 10, 1, 1, 1, 4
start = time.time()
wc, theory = inTOS.wc_tos(mu, L1, L3, alpha, theta, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, n = 1, 5
gamma = 1 / L
start = time.time()
wc, theory = ncGD.wc_gd(L, gamma, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, n = 1, 5
gamma = 1 / L / 2
start = time.time()
wc, theory = ncNL1.wc_no_lips1(L, gamma, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, n = 1, 3
gamma = 1 / L
start = time.time()
wc, theory = ncNL2.wc_no_lips2(L, gamma, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu, n = 1, 0.1, 5
start = time.time()
wc, theory = inSAGA.wc_saga(L, mu, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu, v, R, n = 1, 0.1, 1, 2, 5
gamma = 1 / L
start = time.time()
wc, theory = inSGDSC.wc_sgd(L, mu, gamma, v, R, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu, R, n = 1, 0.1, 2, 5
gamma = 1 / L
start = time.time()
wc, theory = inSGD.wc_sgdo(L, mu, gamma, R, n)
end = time.time()
print('Timing:', end - start, '[s]')

alpha, n = 2, 10
start = time.time()
wc, theory = opAPP.wc_ppm(alpha, n)
end = time.time()
print('Timing:', end - start, '[s]')

alpha, n = 2, 3
start = time.time()
wc, theory = opPPM.wc_ppm(alpha, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu, alpha, theta = 1, 0.1, 1.3, 0.9
start = time.time()
wc, theory = opDRS.wc_drs(L, mu, alpha, theta)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu, beta, alpha, theta = 1, 0.1, 1, 1.3, 0.9
start = time.time()
wc, theory = opTOS.wc_tos(L, mu, beta, alpha, theta)
end = time.time()
print('Timing:', end - start, '[s]')

L, n = 1, 10
start = time.time()
wc, theory = inHI.wc_halpern(L, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, n = 1, 10
start = time.time()
wc, theory = inKM.wc_km(L, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, n = 1, 10
gamma = 1 / L
start = time.time()
wc, theory = potGD1.wc_gd_lyapunov_1(L, gamma, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, n = 1, 10
gamma = 1 / L
start = time.time()
wc, theory = potGD2.wc_gd_lyapunov_2(L, gamma, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, lam = 1, 10
gamma = 1 / L
start = time.time()
wc, theory = potFGD.wc_gd_lyapunov(L, gamma, lam)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu = 1, 0.1
gamma = 1 / L
start = time.time()
wc, theory = inPS1.wc_ps_1(L, mu, gamma)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu = 1, 0.1
gamma = 1 / L
start = time.time()
wc, theory = inPS2.wc_ps_2(L, mu, gamma)
end = time.time()
print('Timing:', end - start, '[s]')
