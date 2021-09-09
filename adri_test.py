import Examples.A_MethodsForUnconstrainedConvexMinimization.OptimizedGradientMethod as OGM
import time

L,n=1,4

start = time.time()
wc,theory = OGM.wc_ogm(L, n)
end = time.time()
print('Timing:', end - start, '[s]')

