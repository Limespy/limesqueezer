import numpy as np
# import matplotlib.pyplot as plt
# import time
# import API



###═════════════════════════════════════════════════════════════════════
def fastcompress(x, y, atol=1e-5, mins = 100):
    '''Fast compression using sampling and splitting from largest error
    x: 1D numpy array
    y: 1D numpy array
    atol: absolute error tolerance
    mins: minimum number of samples, don't change if you don't understand
    '''
    def r(a, b):
        '''Recurser'''
        n = b-a-1
        step = 1 if n<=mins*2 else round(n / (2*(n - mins)**0.5 + mins))

        e = lambda xf, yf: np.abs((y[b]- y[a]) /(x[b] - x[a])* (xf - x[a]) + y[a] - yf)
        i = a + step*np.argmax(e(x[a+1:b-1:step], y[a+1:b-1:step]))

        return np.concatenate((r(a,i), r(i,b)[1:])) if e(x[i], y[i]) > atol else (a,b)

    return r(0,len(x)-1)