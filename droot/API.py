#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Root finding algorithms for strictly ascending discrete functions"""
import math

import time
import sys
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np

from numpy.polynomial import polynomial as poly

def fit(*args):
    return poly.Polynomial.fit(*args)

class F2Zero():
    def __init__(self,n_data,b,atol):
        self.n_data = n_data
        self.b = b
        self.atol = atol
        self.x_data = np.linspace(0,1,int(self.n_data))
        self.y_data = self.reference(self.x_data,self.b)
    
    def reference(self, x,b):
        return b*((1-b)/(b-x) + 1)
    
    def __call__(self,n):
        n = int(n) + 2
        y = self.y_data[:n]
        x = self.x_data[:n]
        return max(abs(y-fit(x, y, 1)(x))) - self.atol
    
    def test_reach(self,reach,is_plot = False):
        t_start = time.perf_counter()
        x1, y1, x2, y2 = reach(self,self.atol)
        t = time.perf_counter()-t_start
        if is_plot: plot(x1,x2)
        return t , x1, y1, x2, y2

    def test_interval(self,interval,x1,y1,x2,y2,isplot=False):
        t_start = time.perf_counter()
        x = interval(self,x1,y1,x2,y2)
        t = time.perf_counter()-t_start
        if is_plot: plot(x1,x2)
        return t , x
###═════════════════════════════════════════════════════════════════════
def reach1(f, y0):
    """Finds the upper limit to interval
    Free 2nd degree polynomial"""

    y1 = f(1)
    if y1 > 0:
        return 0, y0, 1, y1
    else:
        #print(y1)

        # Next step at 3
        y2 = f(3)
        if y2>0:
            return 1, y1, 3, y2
        
        y3 = f(7)
        if y3>0:
            return 3, y2, 7, y3

        xp1 = math.ceil(max(fit([0,1,3,7], [y0,y1,y2,y3], 2).roots()))

        #print(xp1)
        yp1 = f(xp1)

        xp2 = math.ceil(max(fit([0,1,3,7,xp1], [y0,y1,y2,y3,yp1], 2).roots()))
        #print("xp2",xp2)

        if yp1 > 0:
            return 7, y3, xp1, yp1
        
        xp2 = math.ceil(max(fit([3,7,xp1], [y2,y3,yp1], 2).roots()))
        #print(xp2)
        yp2 = f(xp2)

    return xp1, yp1, xp2, yp2
###═════════════════════════════════════════════════════════════════════
def reach2(f, y1, x2):
    """Finds the upper limit to interval
    Exponential"""

    x1 = 0
    y2 = f(x2)
    while y2 < 0:
        
        x1, y1 = x2, y2
        x2 = x1*2
        y2 = f(x2)
        #print(y2)
    return x1, y1, x2, y2
###═════════════════════════════════════════════════════════════════════
def reach3(f, y0, x2):
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
    y2 = f(x2)
    while y2 < 0:
        x1, y1 = x2, y2
        x2 = round(x2 / (1-y2/(y0)+1)**0.5)
        y2 = f(x2)
    return x1, y1, x2, y2
###═════════════════════════════════════════════════════════════════════
def interval1(f,x1,y1,x2,y2):
    """Bisection method"""
    
    xdiff = x2-x1
    if xdiff > 2:
        xn = round((x1+x2)/2) # Linear estimation
        # print(xn)
        yn = f(xn)
        if yn >0:
            x2, y2 = xn, yn
        else: 
            x1, y1 = xn, yn
        return interval1(f,x1,y1,x2,y2)
    elif xdiff == 2:
        yn = f(x1+1)
        return x1 if yn >0 else x1+1
    else:
        return x1
###═════════════════════════════════════════════════════════════════════
def interval2(f,x1,y1,x2,y2):
    """Returns the last x where f(x)<0
    lerp"""

    xdiff = x2-x1
    if xdiff > 2:
        xn = round(x1-y1/(y2-y1)*(x2-x1)) # Linear estimation
        # print(xn)
        if xn == x1:    # To stop repetition in close cases
            xn += 1
        elif xn == x2:
            xn -= 1

        yn = f(xn)
        if yn >0:
            x2, y2 = xn, yn
        else: 
            x1, y1 = xn, yn
        return interval2(f,x1,y1,x2,y2)

    elif xdiff == 2:
        yn = f(x1+1)
        return x1 if yn >0 else x1+1
    else:
        return x1
###═════════════════════════════════════════════════════════════════════
def interval3(f,x1,y1,x2,y2):
    """Returns the last x where f(x)<0
    lerp with refinement"""
    xdiff = x2-x1
    if xdiff > 2:
        xn = round(x1-y1/(y2-y1)*(x2-x1)) # Linear estimation
        if xn == x1:    # To stop repetition in close cases
            xn += 1
        elif xn == x2:
            xn -= 1

        yn = f(xn)

        # Refining the estimation
        xn1 = x1-y1/(yn-y1)*(xn-x1)
        xn2 = xn-yn/(y2-yn)*(x2-xn)
        # print(xn)
        # print(xn1)
        # print(xn2)
        xnr = round((xn1+xn2)/2)
        # print(xnr)

        if xnr > x2 or xnr<x1: # If refinenment fails, it is skipped
            if ynr >0:
                x2, y2 = xn, yn
            else: 
                x1, y1 = xn, yn
            return interval3(f,x1,y1,x2,y2)
        
        ynr = f(xnr)
        
        if yn < 0:
            if ynr < 0: # Both negative
                x1, y1 = (xnr, ynr) if (yn < ynr) else (xn, yn) # Bigger is used
            else:
                x1, y1, x2, y2 = xn, yn, xnr, ynr

        else:
            if ynr > 0: # Both positive
                x2, y2  = (xnr, ynr) if (yn > ynr) else (xn, yn) # Smaller is used
            else:
                x1, y1, x2, y2 = xnr, ynr, xn, yn
        return interval3(f,x1,y1,x2,y2)

    elif xdiff == 2:
        yn = f(x1+1)
        return x1 if yn >0 else x1+1
    else:
        return x1
###═════════════════════════════════════════════════════════════════════
def best(function,y1):
    x1, y1, x2, y2 = reach3(function,y1)
    return interval2(x1, y1, x2, y2)
