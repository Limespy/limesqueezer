#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy import interpolate
import matplotlib.pyplot as plt
import time
from numpy.polynomial import polynomial as poly
import numba
import sys
def lfit(*args):
    return poly.Polynomial.fit(*args)

class Data():
    def __init__(self,x_data=None,y_data=None,n_data=None, b = 3):
        if x_data:
            self.x_data = x_data
            self.n_data = len(self.x_data)
            self.y_data = y_data if y_data else self.reference(self.x_data,b)
        else:
            self.n_data = int(1e5) if not n_data else int(n_data)
            self.x_data = np.linspace(0,1,int(self.n_data))
            self.y_data = self.reference(self.x_data,b)

        self.x_compressed = []
        self.y_compressed = []
    #─────────────────────────────────────────────────────────────────── 
    def reference(self, x,c):
        # Setting up the reference data
        #return 2 - x/ (1+c-c*x)
        return np.sin(x*c*4)+3
    #───────────────────────────────────────────────────────────────────
    def compress(self,atol=1e-5, mins=30):
        """Compresses the data"""

        
        t_start = time.perf_counter()

        x_slice = self.x_data
        y_slice = self.y_data
        limit = self.n_data - 2
        self.atol = atol
        errscale = np.max(np.abs(
                                 (y_slice[-1]- y_slice[0])
                                /(x_slice[-1] - x_slice[0])
                                * (x_slice[0:-1:int(limit/10)]-x_slice[0])
                                + y_slice[0] - y_slice[0:-1:int(limit/10)]
                                )) / self.atol
        ndiv = 2*errscale**0.5 + 1
        estimate = int(limit/(ndiv-len(self.x_compressed)))
        # print("estimate", estimate)

        #───────────────────────────────────────────────────────────────
        def initial_f2zero(n):
            n = int(n) + 2
            step = 1 if n<=mins else round(n/(2*(n - mins)**0.5 + mins))
            x = x_slice[0:n:step]
            y = y_slice[0:n:step]
            fit = lfit(x, y, 1)
            return max(np.abs(fit(x)-y)) - self.atol, fit
        #───────────────────────────────────────────────────────────────
        y_range = max(y_slice) - min(y_slice)
        #estimate = int(limit * (self.atol/y_range)**0.5) + 1
        n2, fit = droot(initial_f2zero, - self.atol, estimate, limit)
        self.x_compressed.append(0)
        self.y_compressed.append(fit(0))
        #───────────────────────────────────────────────────────────────
        
        def f2zero(n,xs, ys, x0, y0,atol):
            n = int(n)+1
            step = 1 if n<=mins else round(n/(2*(n - mins)**0.5 + mins))
            a = (ys[n] - y0) / (xs[n] - x0)
            b = y0 - a * x0
            fit = lambda x: a*x + b
            return max(np.abs(fit(xs[0:n:step])- ys[0:n:step]))-atol , fit
        #───────────────────────────────────────────────────────────────
        
        while n2 < limit :
            x1 = (x_slice[n2] + x_slice[n2+1])/2
            self.x_compressed.append(x1)
            y1 = fit(self.x_compressed[-1])
            self.y_compressed.append(y1)
            x_slice = x_slice[n2+1:]
            y_slice = y_slice[n2+1:]

            limit = limit - n2 - 1
            #scaler = 
            # print(limit*(self.atol/(max(y_slice) - min(y_slice)))**0.5)
            scaler = ndiv-len(self.x_compressed)
            estimate = int(limit/scaler)+1 if scaler>1 else limit

            # print(limit)
            #print(scaler)
            # print("estimate", estimate)

            n2, fit = droot(lambda n: f2zero(n, x_slice, y_slice, x1, y1,self.atol ),
                            -self.atol, estimate, limit)
            # print("n2",n2)

        self.x_compressed.append(x_slice[-1])
        self.y_compressed.append(fit(self.x_compressed[-1]))

        self.x_compressed = np.array(self.x_compressed)
        self.y_compressed = np.array(self.y_compressed)
        t = time.perf_counter()-t_start
        print("Compression time\t%.3f ms" % (t*1e3))
        print("Length of compressed array\t%i"%len(self.x_compressed))
        compression_factor = 1 - len(self.x_compressed)/len(self.x_data)
        print("Compression factor\t%.3f %%" % (compression_factor*1e2))
    #───────────────────────────────────────────────────────────────
    def simplecompress(self,atol=1e-5, mins=30):

        t_start = time.perf_counter()

        x_slice = self.x_data
        y_slice = self.y_data
        limit = self.n_data - 2
        self.atol = atol
        errscale = np.max(np.abs(
                                 (y_slice[-1]- y_slice[0])
                                /(x_slice[-1] - x_slice[0])
                                * (x_slice[0:-1:int(limit/10)]-x_slice[0])
                                + y_slice[0] - y_slice[0:-1:int(limit/10)]
                                )) / self.atol
        ndiv = 2*errscale**0.5 + 1
        #───────────────────────────────────────────────────────────────
        def f2zero(n,xs, ys,atol):
            n = int(n)+2
            step = 1 if n<=mins else round(n/(2*(n - mins)**0.5 + mins))
            a = (ys[n] - ys[0])/(xs[n] - xs[0])
            b = ys[0] - a * xs[0]
            return max(np.abs(a*xs[0:n:step]+ b - ys[0:n:step]))-atol , None
        #───────────────────────────────────────────────────────────────
        n2 = -1
        while n2 < limit:
            self.x_compressed.append(x_slice[n2+1])
            self.y_compressed.append(y_slice[n2+1])
            x_slice = x_slice[n2+1:]
            y_slice = y_slice[n2+1:]

            limit = limit - n2 - 2
            scaler = ndiv-len(self.x_compressed)
            estimate = int(limit/scaler)+1 if scaler>1 else limit
            # print(limit)
            # print(scaler)
            # print("estimate", estimate)

            n2, _ = droot(lambda n: f2zero(n, x_slice, y_slice, self.atol),
                            -self.atol, estimate, limit)
            # print("n2", n2)
        #───────────────────────────────────────────────────────────────
        self.x_compressed.append(x_slice[-1])
        self.y_compressed.append(y_slice[-1])


        self.x_compressed = np.array(self.x_compressed)
        self.y_compressed = np.array(self.y_compressed)
        t = time.perf_counter()-t_start
        print("Compression time\t%.3f ms" % (t*1e3))
        print("Length of compressed array\t%i"%len(self.x_compressed))
        compression_factor = 1 - len(self.x_compressed)/len(self.x_data)
        print("Compression factor\t%.3f %%" % (compression_factor*1e2))
    #───────────────────────────────────────────────────────────────
    def fastcompress(self, atol=1e-5, mins = 30):
        
        # t_start = time.perf_counter()
        self.atol = atol
        def rec(a, b):
            n = b-a-1
            step = 1 if n<=mins else round(n / (2*(n - mins)**0.5 + mins))

            x1, y1 = self.x_data[a], self.y_data[a]
            x2, y2 = self.x_data[b], self.y_data[b]
            # print(x1,y1,x2,y2)
            err = lambda x, y: np.abs((y2- y1) /(x2 - x1)* (x - x1) + y1 - y)
            i = a + step*np.argmax(err(self.x_data[a:b:step], self.y_data[a:b:step]))
            # print(index)
            # time.sleep(0.5)
            # print(a,b)
            # print(err(self.x_data[index], self.y_data[index]))
            return np.concatenate((rec(a, i), rec(i, b)[1:])) if err(self.x_data[i], self.y_data[i]) > atol else [a,b]
        indices= rec(0,self.n_data-1)

        self.x_compressed = self.x_data[indices]
        self.y_compressed = self.y_data[indices]
        # t = time.perf_counter()-t_start
        # print("Compression time\t%.3f ms" % (t*1e3))
        # print("Length of compressed array\t%i"%len(self.x_compressed))
        # compression_factor = 1 - len(self.x_compressed)/len(self.x_data)
        # print("Compression factor\t%.3f %%" % (compression_factor*1e2))

    #───────────────────────────────────────────────────────────────────
    def make_lerp(self):
        self.lerp = interpolate.interp1d(self.x_compressed,
                                         self.y_compressed,
                                         assume_sorted=True)
    #───────────────────────────────────────────────────────────────────
    def residual(self):
        """Error residuals"""
        return self.lerp(self.x_data) - self.y_data


# Given atol and Delta_y, 
# in the best case 1 line would be enough 
# and in the worst case Delta_y / atol.
#
# Geometric mean between these would maybe be good choice,
# so likely around n_lines ~ sqrt(Delta_y / atol)
# meaning delta_x ~ Delta_x * sqrt(atol / Delta_y)
# 
# When this is normalised so Delta_y = 1 and Delta_x = n,
# delta_x ~ n * sqrt(atol)

###═════════════════════════════════════════════════════════════════════
def interval2(f,x1,y1,x2,y2):
    """Returns the last x where f(x)<0
    lerp"""
    
    while x2 - x1 > 2:

        xn = round((x1-y1/(y2-y1)*(x2-x1) + (x2+ x1)/2)/2) # Getting best fit
        # Linear estimation

        if xn == x1:    # To stop repetition in close cases
            xn += 1
        elif xn == x2:
            xn -= 1

        yn, fit = f(xn)
        if yn > 0:
            x2, y2 = xn, yn
        else: 
            x1, y1 = xn, yn

    if x2 - x1 == 2:

        yn, fit = f(x1+1)
        return (x1, fit) if yn >0 else (x1+1, fit)
    else:

        _, fit = f(x1)
        return (x1, fit)
###═════════════════════════════════════════════════════════════════════
def droot(f, y0, x2, limit):
    """Finds the upper limit to interval
    Limited 2nd degree polynomial
    
    Prediction heuristic 2
        Function is approximately a*x^2 + y0
        Fitting a*x1^2 + y0 = y1 -> a = (y1-y0)/x1^2
        Root a*x^2 +y0 = 0 -> x = sqrt(-y0/a) 
        Root a*xp^2 +y0 = 0 -> 
        xp = sqrt(-y0/((y1-y0)/x1^2)) = x1 * sqrt(-y0/(y1-y0))
        = x1 / sqrt(1-y1/y0) 
        """
    # x2 *=4
    x1 = 0
    y1 = y0
    y2, fit = f(x2)
    x2e = x2
    while y2 < 0:
        # time.sleep(0.5)
        x1, y1 = x2, y2
        # # print(x1,x2,y1,y2)
        # x2l = int(x2 /(1-y2/y0)**4)+1
        # # print("lin",x2l)
        # if x2l < limit:
        #     y2l, _ = f(x2l)
        #     # print("lerp",x2l,y2l)
        
        # #x2e = 2*x2 # int((x2 + x2 / (1-y2/(y0))**0.5)/2)+1
        # # print("exp",x2e)
        # #x2p = int(x2 /(1-y2/y0)**0.5)+2
        x2 *= 2
        if x2 >= limit:
            y2, fit = f(limit)
            if y2<0:
                return limit, fit
            else:
                x2 = limit
                break
        # if x2e < limit:
        #     y2e, _ = f(x2e)
        # #     print("yexp",y2e)
        # # print("poly 2",x2p)
        # if x2p<limit:
        #     y2p, _ = f(x2p)
        #     # print(y2p)
        y2, fit = f(x2)
        
    return interval2(f,x1, y1, x2, y2)
###═════════════════════════════════════════════════════════════════════

n_data = int(float(sys.argv[1]))
atol = float(sys.argv[2])
mins = int(float(sys.argv[3]))
b = int(float(sys.argv[4]))
# data = Data(n_data=n_data,b=b)

# data.compress(atol=atol,mins = mins)

# plt.figure()
# plt.plot(data.x_data,data.y_data)
# plt.plot(data.x_compressed,data.y_compressed,"-o")

# # data.make_lerp()
# # plt.figure()
# # tol = abs(data.residual())-data.atol
# # plt.plot(data.x_data,tol)

# data2 = Data(n_data=n_data,b=b)

# data2.simplecompress(atol=atol,mins = mins)

# plt.figure()
# plt.plot(data2.x_data,data2.y_data)
# plt.plot(data2.x_compressed,data2.y_compressed,"-o")


data3 = Data(n_data=n_data,b=b)

t_start = time.perf_counter()
data3.fastcompress(atol=atol, mins = mins)
xce, yce = data3.x_compressed, data3.y_compressed
t = time.perf_counter()-t_start
print("Compression time\t%.3f ms" % (t*1e3))
print("Length of compressed array\t%i"%len(xce))
compression_factor = 1 - len(xce)/len(data3.x_data)
print("Compression factor\t%.3f %%" % (compression_factor*1e2))


plt.figure()
plt.plot(data3.x_data,data3.y_data)
plt.plot(data3.x_compressed,data3.y_compressed,"-o")
data3.make_lerp()
tol = abs(data3.residual())-data3.atol
print(max(tol))
plt.figure()
plt.plot(data3.x_data,tol)

def fastcompress(x,y, atol=1e-5, mins = 100):
    """Fast compression using sampling and splitting from largest error"""

    def rec(a, b):
        """Recurser"""
        n = b-a-1
        step = 1 if n<=mins else round(n / (2*(n - mins)**0.5 + mins))

        err = lambda xf, yf: np.abs((y[b]- y[a]) /(x[b] - x[a])* (xf - x[a]) + y[a] - yf)
        i = a + step*np.argmax(err(x[a+1:b-1:step], y[a+1:b-1:step]))

        return np.concatenate((rec(a, i), rec(i, b)[1:])) if err(x[i], y[i]) > atol else (a,b)

    return rec(0,len(x)-1)

t_start = time.perf_counter()
indices = fastcompress(data3.x_data, data3.y_data, atol=atol, mins = mins)
data3.x_compressed, data3.y_compressed = data3.x_data[indices], data3.y_data[indices]
t = time.perf_counter()-t_start
print("Compression time\t%.3f ms" % (t*1e3))
print("Length of compressed array\t%i"%len(xce))
compression_factor = 1 - len(xce)/len(data3.x_data)
print("Compression factor\t%.3f %%" % (compression_factor*1e2))

data3.make_lerp()
tol = abs(data3.residual())-data3.atol
print(max(tol))
plt.figure()
plt.plot(data3.x_data,tol)

# c = 0.5
# n = 30
# x = np.linspace(0,1,n)
# xm = 2*x-1
# y = xm*((1-c)*xm**2+c)/2+0.5
# plt.figure()
# plt.plot(x,y,"o")
# plt.plot(y,np.zeros(n),"o")

# density = 1/np.diff(y)
# dx = x[:-1] + np.diff(x)/2
# plt.figure()
# plt.plot(dx,density/(min(density)),"o")

plt.show()