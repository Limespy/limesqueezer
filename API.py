#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy import interpolate
import matplotlib.pyplot as plt
import time
from numpy.polynomial import polynomial as poly

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

        self.x_compressed = [0]
        self.y_compressed = []
    #─────────────────────────────────────────────────────────────────── 
    def reference(self, x,c):
        # Setting up the reference data
        return 2 - x/ (1+c-c*x)
        # return np.sin(x*c*4)+3
    #───────────────────────────────────────────────────────────────────
    def compress(self,atol=1e-5):
        """Compresses the data"""
        t_start = time.perf_counter()

        x_slice = self.x_data
        y_slice = self.y_data
        limit = self.n_data - 2
        self.atol = atol
        #───────────────────────────────────────────────────────────────
        def initial_f2zero(n):
            n = int(n) + 2
            x = x_slice[:n]
            y = y_slice[:n]
            fit = lfit(x, y, 1)
            return max(abs(fit(x)-y)) - self.atol, fit
        #───────────────────────────────────────────────────────────────
        y_range = max(y_slice) - min(y_slice)
        estimate = int(limit * (self.atol/y_range)**0.5) + 1
        n2, fit = droot(initial_f2zero, - self.atol, estimate, limit)
        
        self.y_compressed.append(fit(0))
        #───────────────────────────────────────────────────────────────
        
        def f2zero(n,xs, ys, x0, y0):
            n = int(n)+1
            xd = xs[:n]
            yd = ys[:n]
            a = (ys[n] - y0)/(xs[n] - x0)
            b = y0 - a * x0
            fit = lambda x: a*x + b
            return max(abs(fit(xs[:n])- ys[:n]))-self.atol , fit
        #───────────────────────────────────────────────────────────────
        
        while n2 < limit :
            x1 = (x_slice[n2] + x_slice[n2+1])/2
            self.x_compressed.append(x1)
            y1 = fit(self.x_compressed[-1])
            self.y_compressed.append(y1)
            x_slice = x_slice[n2+1:]
            y_slice = y_slice[n2+1:]

            limit = limit - n2 - 1
            scaler = (self.atol/(max(y_slice) - min(y_slice)))**0.5
            estimate = int(limit*scaler) if scaler<1 else limit

            n2, fit = droot(lambda n: f2zero(n, x_slice, y_slice, x1, y1),
                            -self.atol, estimate, limit)

        self.x_compressed.append(x_slice[-1])
        self.y_compressed.append(fit(self.x_compressed[-1]))

        self.x_compressed = np.array(self.x_compressed)
        self.y_compressed = np.array(self.y_compressed)
        t = time.perf_counter()-t_start
        print("Compression time\t%.3f s" % t)
        print(len(self.x_compressed))
        compression_factor = 1 - len(self.x_compressed)/len(self.x_data)
        print("Compression factor\t%.3f %%" % (compression_factor*1e2))
    #───────────────────────────────────────────────────────────────
    def simplecompress(self,atol=1e-5):

        t_start = time.perf_counter()

        x_slice = self.x_data
        y_slice = self.y_data
        limit = self.n_data - 2
        self.atol = atol
        #───────────────────────────────────────────────────────────────
        def f2zero(n,xs, ys):
            n = int(n)+2
            a = (ys[n] - ys[0])/(xs[n] - xs[0])
            b = ys[0] - a * xs[0]
            fit = lambda x: a*x + b
            return max(abs(fit(xs[:n])- ys[:n]))-self.atol , fit
        #───────────────────────────────────────────────────────────────
        n2 = -1
        while n2 < limit :
            self.x_compressed.append(x_slice[n2+1])
            self.y_compressed.append(y_slice[n2+1])
            x_slice = x_slice[n2+1:]
            y_slice = y_slice[n2+1:]

            y_range = max(y_slice) - min(y_slice)

            limit = limit - n2 - 2

            estimate = int(limit * (self.atol/y_range)**0.5)+1
        

            n2, fit = droot(lambda n: f2zero(n, x_slice, y_slice),
                            -self.atol, estimate, limit)
        #───────────────────────────────────────────────────────────────
        self.x_compressed.append(x_slice[-1])
        self.y_compressed.append(fit(self.x_compressed[-1]))


        self.x_compressed = np.array(self.x_compressed)
        self.y_compressed = np.array(self.y_compressed)
        t = time.perf_counter()-t_start
        print("Compression time\t%.3f s" % t)
        print(len(self.x_compressed))
        compression_factor = 1 - len(self.x_compressed)/len(self.x_data)
        print("Compression factor\t%.3f %%" % (compression_factor*1e2))

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

def droot(f, y0, x2, limit):
    """Finds the upper limit to interval
    Limited 2nd degree polynomial
    
    Prediction heuristic 2
        Function is approximately a*x^2 - atol
        Fitting a*x1^2 - atol = y1 -> a = (y1+atol)/x1^2
        Root a*x^2 -atol = 0 -> x = sqrt(atol/a) 
        Root a*xp^2 -atol = 0 -> 
        xp = sqrt(atol/((y1+atol)/x1^2)) = x1 * sqrt(atol/(y1+atol))
        = x1 / sqrt(y1/atol+1) 
        """
    x1 = 0
    y1 = y0

    y2, fit = f(x2)
    while y2 < 0:
        x1, y1 = x2, y2
        x2 *= 2 # int((x2 + x2 / (1-y2/(y0))**0.5)/2)+1
        if x2 >= limit:
            y2, fit = f(limit)
            if y2<0:
                return limit, fit
            else:
                x2 = limit
                break
        y2, fit = f(x2)
    return interval2(f,x1, y1, x2, y2)
###═════════════════════════════════════════════════════════════════════
data = Data(n_data=1e5,b=10)

data.compress(atol=1e-3)

plt.figure()
plt.plot(data.x_data,data.y_data)
plt.plot(data.x_compressed,data.y_compressed,"-o")

data.make_lerp()
data.residual
plt.figure()

tol = abs(data.residual())-data.atol

plt.plot(data.x_data,tol)

data2 = Data(n_data=1e5,b=10)

data2.simplecompress(atol=1e-3)


plt.show()