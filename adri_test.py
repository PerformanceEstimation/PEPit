import Examples.A_MethodsForUnconstrainedConvexMinimization.OptimizedGradientMethod as OGM
import Examples.A_MethodsForUnconstrainedConvexMinimization.GradientExactLineSearch as ELS
import Examples.A_MethodsForUnconstrainedConvexMinimization.Subgradient as SG
import numpy as np

import time

L,n=1,4

start = time.time()
wc,theory = OGM.wc_ogm(L, n)
end = time.time()
print('Timing:', end - start, '[s]')


L,mu,n=1,.001,2

start = time.time()
wc,theory = ELS.wc_ELS(L=L, mu=mu, n=n)
end = time.time()
print('Timing:', end - start, '[s]')


M,N=2,50
gamma = 1/(np.sqrt(N+1)*M)

start = time.time()
wc,theory =  SG.wc_subgd(M=M, N=N, gamma=gamma)
end = time.time()
print('Timing:', end - start, '[s]')



