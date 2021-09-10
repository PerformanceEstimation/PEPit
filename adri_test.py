import Examples.A_MethodsForUnconstrainedConvexMinimization.OptimizedGradientMethod as OGM
import Examples.A_MethodsForUnconstrainedConvexMinimization.GradientExactLineSearch as ELS
import Examples.A_MethodsForUnconstrainedConvexMinimization.Subgradient as SG
import Examples.A_MethodsForUnconstrainedConvexMinimization.ConjugateGradientMethod as CG
import Examples.A_MethodsForUnconstrainedConvexMinimization.InexactGradientExactLineSearch as inELS
import Examples.A_MethodsForUnconstrainedConvexMinimization.InexactGradient as inGD
import Examples.A_MethodsForUnconstrainedConvexMinimization.InexactAcceleratedGradient as inAGM

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


L,mu,epsilon,n=1,.1,.1,2

start = time.time()
wc,theory = inELS.wc_InexactGrad_ELS(L=L, mu=mu, epsilon=epsilon, n=n)
end = time.time()
print('Timing:', end - start, '[s]')


L,mu,epsilon,n=1,.1,.1,2

start = time.time()
wc,theory = inGD.wc_InexactGrad(L=L, mu=mu, epsilon=epsilon, n=n)
end = time.time()
print('Timing:', end - start, '[s]')


L,epsilon,n=1,0,5

start = time.time()
wc,theory = inAGM.wc_InexactAGM(L=L, epsilon=epsilon, n=n)
end = time.time()
print('Timing:', end - start, '[s]')


L,epsilon,n=1,.1,5

start = time.time()
wc,theory = inAGM.wc_InexactAGM(L=L, epsilon=epsilon, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

