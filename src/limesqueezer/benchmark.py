'''Running a benchmark'''
import numpy as np
import sys
import API as ls
import time
# ls._G['profiling'] = True

xdata = np.linspace(0,10,int(1e5))
ydata = np.array([np.sin(xdata*xdata), np.sin(xdata*1.5*xdata)]).T

starttime = time.time()
endtime = starttime + 100
use_numba = int('--numba' in sys.argv)
n = 0
while time.time() < endtime:
    n += 1
    print(n)
    for _ in range(10):
        ls(xdata, ydata, tol = 1e-3, use_numba = use_numba)
print(n)

