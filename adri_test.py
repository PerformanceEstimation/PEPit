import Examples.A_MethodsForUnconstrainedConvexMinimization.OptimizedGradientMethod as OGM
import Examples.A_MethodsForUnconstrainedConvexMinimization.GradientExactLineSearch as ELS

import time

L,n=1,4

start = time.time()
wc,theory = OGM.wc_ogm(L, n)
end = time.time()
print('Timing:', end - start, '[s]')


L,mu,n=1,.1,2

start = time.time()
wc,theory = ELS.wc_ELS(L=L, mu=mu, n=n)
end = time.time()
print('Timing:', end - start, '[s]')

