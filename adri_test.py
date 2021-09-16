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

import numpy as np

import time

L,n=3,4

start = time.time()
wc,theory = OGM.wc_ogm(L, n)
end = time.time()
print('Timing:', end - start, '[s]')


L,mu,n=3,.1,1

start = time.time()
wc,theory = ELS.wc_ELS(L=L, mu=mu, n=n)
end = time.time()
print('Timing:', end - start, '[s]')


M,N=2,10
gamma = 1/(np.sqrt(N+1)*M)

start = time.time()
wc,theory =  SG.wc_subgd(M=M, N=N, gamma=gamma)
end = time.time()
print('Timing:', end - start, '[s]')


L,n=3,2

start = time.time()
wc,theory = CG.wc_CG(L=L, n=n)
end = time.time()
print('Timing:', end - start, '[s]')


L,mu,epsilon,n=3,.1,.1,2

start = time.time()
wc,theory = inELS.wc_InexactGrad_ELS(L=L, mu=mu, epsilon=epsilon, n=n)
end = time.time()
print('Timing:', end - start, '[s]')


L,mu,epsilon,n=3,.1,.1,2

start = time.time()
wc,theory = inGD.wc_InexactGrad(L=L, mu=mu, epsilon=epsilon, n=n)
end = time.time()
print('Timing:', end - start, '[s]')


L,epsilon,n=3,0,5

start = time.time()
wc,theory = inAGM.wc_InexactAGM(L=L, epsilon=epsilon, n=n)
end = time.time()
print('Timing:', end - start, '[s]')


L,epsilon,n=2,.1,5

start = time.time()
wc,theory = inAGM.wc_InexactAGM(L=L, epsilon=epsilon, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

mu = 0.1
L = 1.
alpha = 4*L/(np.sqrt(L)+np.sqrt(mu))**2
beta = ((np.sqrt(L)-np.sqrt(mu))/(np.sqrt(L)+np.sqrt(mu)))**2
n = 3

start = time.time()

wc = inHBM.wc_heavyball(mu=mu, L=L, alpha=alpha, beta=beta, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

A0, n = 1,3
gammas = [1, 1, 1]

start = time.time()

wc = inFPM.wc_fppa(A0=A0,gammas=gammas,n=n)
end = time.time()
print('Timing:', end - start, '[s]')


L, mu, n = 1,.1,1

start = time.time()

wc = inTMM.wc_tmm(mu=mu, L=L, n=n)
end = time.time()
print('Timing:', end - start, '[s]')


L, mu, n= 1,.1,1
lam = .5

start = time.time()

wc = inRMM.wc_rmm(mu=mu, L=L, lam=lam)
end = time.time()
print('Timing:', end - start, '[s]')

mu,L,n=0, 1, 10
start = time.time()
wc,theory = inFGM.wc_fgm(mu,L, n)
end = time.time()
print('Timing:', end - start, '[s]')

L, mu, n = 1,.1,5

start = time.time()
wc, theory = instrFGM.wc_fgm(mu=mu, L=L, n=n)
end = time.time()
print('Timing:', end - start, '[s]')


mu,L,n=0, 1, 5
start = time.time()
wc,theory = inFPGM.wc_fgm(mu,L, n)
end = time.time()
print('Timing:', end - start, '[s]')
