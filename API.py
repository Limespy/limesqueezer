#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy
import matplotlib.pyplot as plt

from numpy.polynomial import polynomial as poly

def lfit(*args):
    return poly.Polynomial.fit(*args)

class Data():
    def __init__(self,x_data=None,y_data=None,n_data=None, atol=1e-5, b = 3):
        if x_data:
            self.x_data = x_data
            self.n_data = len(self.x_data)
            self.y_data = y_data if y_data else self.reference(self.x_data,b)
        else:
            self.n_data = int(1e5) if not n_data else n_data:
            self.x_data = np.linspace(0,1,int(n_data))
            self.y_data = self.reference(self.x_data,b)
        
        self.atol = atol
        self.x_compressed = [0]
        self.y_compressed = []
    #─────────────────────────────────────────────────────────────────── 
    def reference(self, x,b):
        # Setting up the reference data
        return b*((1-b)/(b-x) + 1)
    #───────────────────────────────────────────────────────────────────
    def err_max_LSQ(self,x_data,y_data):
        fir = lfit(x_data, y_data, 1)
        return max(abs(y_data-fit(x_data))), fit
    #───────────────────────────────────────────────────────────────────
    def err_n(self,n):

        return self.err_max_LSQ(self.x_data[:(n+2)], self.y_data[:(n+2)])
    #───────────────────────────────────────────────────────────────────
    def compress():
        """Compresses the data"""
        #───────────────────────────────────────────────────────────────
        def initial_f2zero(n):
            n = int(n) + 2
            y = self.y_data[:n]
            x = self.x_data[:n]
            return max(abs(y-lfit(x, y, 1)(x))) - self.atol
        #───────────────────────────────────────────────────────────────
        y_range = max(self.y_data) - min(self.y_data)
        n2, fit = droot(initial_f2zero,
                   - self.atol,
                   self.n_data * (self.atol/y_range)**0.5)
        
        self.y_compressed.append(fit(0))
        self.x_compressed.append((self.x_data[n2+1] + self.x_data[n2+2])/2)
        self.y_compressed.append(fit(self.x_compressed[-1]))

        def f2zero(n):
            n = int(n) + 2
            y = self.y_data[:n]
            x = self.x_data[:n]
            np.mean(fit(x, y, 1)(x))
            

    #───────────────────────────────────────────────────────────────────
    def make_lerp(self):
        self.lerp = scipy.interpolate.interp1d(self.x_compressed,
                                                   self.y_compressed)
    #───────────────────────────────────────────────────────────────────
    def residual(self):
        """Error residuals"""
        return self.y_data - self.lerp(self.x_data)

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
def droot(f, y0, x2):
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
        x2 = round(x2 / (1-y2/(y0)+1)**0.5)
        y2, fit = f(x2)
    #───────────────────────────────────────────────────────────────────
    def interval2(f,x1,y1,x2,y2):
        """Returns the last x where f(x)<0
        lerp"""
        xdiff = x2 - x1
        if xdiff > 2:
            xn = round(x1-y1/(y2-y1)*(x2-x1)) # Linear estimation
            # print(xn)
            if xn == x1:    # To stop repetition in close cases
                xn += 1
            elif xn == x2:
                xn -= 1

            yn, fit = f(xn)
            if yn > 0:
                x2, y2 = xn, yn
            else: 
                x1, y1 = xn, yn
            return interval2(f,x1,y1,x2,y2)

        elif xdiff == 2:
            yn, fit = f(x1+1)
            return (x1, fit) if yn >0 else (x1+1, fit)
        else:
            return (x1, fit)
    #───────────────────────────────────────────────────────────────────

    return interval2(f,x1, y1, x2, y2)

compression = Data()

