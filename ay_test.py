# import Examples.A_MethodsForUnconstrainedConvexMinimization.FastGradientMethod_StronglyConvex as FGM
import Examples.A_MethodsForUnconstrainedConvexMinimization.FastGradientMethod as FGM
import time

for n_i in [1,2,4, 8, 16]:

    mu,L,n=0.1, 1, n_i
    print('n  is', n)
    start = time.time()
    wc,theory = FGM.wc_fgm(mu,L, n)
    end = time.time()
    print('Timing:', end - start, '[s]')

