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

path_cwd = pathlib.Path(os.getcwd()).absolute()
path_home = pathlib.Path(__file__).parent.absolute()

path_figures = path_home / "figures"

is_verbose = "--verbose" or "-v" in sys.argv
is_plot = "--plot" in sys.argv
is_save = "--save" in sys.argv
is_show = "--show" in sys.argv

def fit(*args):
    return poly.Polynomial.fit(*args)



n_data = 1e7
def get_f2zero(n,b):
    x_data = np.linspace(0,1,int(n_data))

    y_data = reference(x_data,b)

atol = 1e-5

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

# n_arr = [int(2 ** k) for k in range(20)]
# t_arr = []
# for n in n_arr:
#     t_start = time.perf_counter()
#     f2zero(n)
#     t_arr.append(time.perf_counter()-t_start)

# plt.figure()
# plt.plot(n_arr, t_arr, marker="o")

# if is_show: plt.show()
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
def reach2(f, y1):
    """Finds the upper limit to interval
    Exponential"""

    x1 = 0
    
    x2 = round(f.n_data * atol**0.5)
    y2 = f(x2)
    
    while y2 < 0:
        
        x1, y1 = x2, y2
        x2 = x1*2
        y2 = f(x2)
        #print(y2)
    return x1, y1, x2, y2
###═════════════════════════════════════════════════════════════════════
def reach3(f, y1):
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
    x2 = round(f.n_data * atol**0.5)
    y2 = f(x2)
    while y2 < 0:
        x1, y1 = x2, y2
        x2 = round(x2 / (y2/f.atol+1)**0.5)
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
def droot(f,y1):
    x1, y1, x2, y2 = reach3(f,y1)
    return interval2(x1, y1, x2, y2)

# 3 starting divisions with ratio r = sum/first > 1
# solve x^2 + x + 1 - r = 0
#
# x = sqrt(1/4 - (1-r)) - 1/2
# O
# first = sum / r
# second = first*(1+x) = first*(sqrt(0.75 + r)) + 0.5)
# third = sum

def slow(f):
    x = 0
    y = f(1)
    while y < 0:
        x += 1
        y = f(x)
    return x-1

def plot(x1,x2):
    n_arr = np.arange(x1,x2+1)
    plt.figure()
    plt.plot(n_arr, [f2zero(n) for n in n_arr])
    plt.plot([x1,x2],[0,0])

f2zero = F2Zero(1e5,1.5,1e-4)
# print("fzero(n=0)",f2zero(0))

def looptest(reach,plot=False):
    b_arr = np.linspace(1.05,1.7,20)
    t_arr = []
    x1_arr = []
    x2_arr = []
    for b in b_arr:
        f2zero = F2Zero(1e5,b,1e-4)
        t , x1, y1, x2, y2 = f2zero.test_reach(reach)
        t_arr.append(t)
        x1_arr.append(x1)
        x2_arr.append(x2)
    print(np.mean(t_arr))
    print(np.mean(np.array(x2_arr) - np.array(x1_arr)))
    return t, x1_arr, x2_arr

# t2, _, _ = looptest(reach2)
# t3, _, _ = looptest(reach3)

# print(np.mean(t3)/np.mean(t2))
# print("Fraction")
# print(f2zero.test_reach(reach3))

t_reach , x1, y1, x2, y2 = f2zero.test_reach(reach3)
print("reach\t\t", t_reach*1e3, x1, x2)

t_interval, x = f2zero.test_interval(interval1, x1, y1, x2, y2)
print("interval\t", (t_reach+t_interval)*1e3, x)

t_interval, x = f2zero.test_interval(interval2, x1, y1, x2, y2)
print("interval\t", (t_reach+t_interval)*1e3, x)

t_interval, x = f2zero.test_interval(interval3, x1, y1, x2, y2)
print("interval\t", (t_reach+t_interval)*1e3, x)




# t_start = time.perf_counter()
# x= slow(f2zero)
# t = time.perf_counter()-t_start
# print("slow", t*1e3, x)

if is_show: plt.show()

