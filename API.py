#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import time
from numpy.polynomial import polynomial as poly
import numba

def lfit(*args):
    return poly.Polynomial.fit(*args)
###═════════════════════════════════════════════════════════════════════
class Data():
    '''Data container'''
    def __init__(self,x=None,y=None,n_data=None, b = 3):
        if x:
            self.x = x
            self.n_data = len(self.x)
            self.y = y if y else self.reference(self.x,b)
        else:
            self.n_data = int(1e5) if not n_data else int(n_data)
            self.x = np.linspace(0,1,int(self.n_data))
            self.y = self.reference(self.x,b)
        
        self.y_range = np.max(self.y)-np.min(self.y)

        self.x_compressed = None
        self.y_compressed = None
    #─────────────────────────────────────────────────────────────────── 
    def reference(self, x,c):
        # Setting up the reference data
        # return 2 - x/ (1+c-c*x)
        return np.sin(x**2*c*2)+3
    #───────────────────────────────────────────────────────────────────
    def make_lerp(self):
        self.lerp = interpolate.interp1d(self.x_compressed, self.y_compressed,
                                            assume_sorted=True)
        self.residuals = self.lerp(self.x) - self.y
        self.NRMSE = np.std(self.residuals)/self.y_range
        self.covariance = np.cov((self.lerp(self.x), self.y))
###═════════════════════════════════════════════════════════════════════
### AUXILIARY FUNCTIONS
def interval2(f,x1,y1,x2,y2):
    '''Returns the last x where f(x)<0'''
    
    while x2 - x1 > 2:
        # Average between linear estimate and half
        xn = int((x1-y1/(y2-y1)*(x2-x1) + (x2 + x1)/2)/2) + 1

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
        yn, fit = f(x2-1)
        return (x2-1, fit) if yn >0 else (x2, fit)
    else:
        _, fit = f(x2)
        return (x2, fit)
###═════════════════════════════════════════════════════════════════════
def droot(f, y0, x2, limit):
    '''Finds the upper limit to interval
    '''
    x1, y1 = 0, y0
    y2, fit = f(x2)

    while y2 < 0:
        x1, y1 = x2, y2
        x2 *= 2
        if x2 >= limit:
            y2, fit = f(limit)
            if y2<0:
                return limit+1, fit
            else:
                x2 = limit
                break
        y2, fit = f(x2)
    return interval2(f,x1, y1, x2, y2)
###═════════════════════════════════════════════════════════════════════
# @numba.jit(nopython=True,cache=True)
def n_lines(x,y,x0,y0,step,atol):
    '''Estimates number of lines required to fit within error tolearance'''
    errscale = 0.5*np.max(np.abs((y[-1]- y0)/(x[-1] - x0)* (x[:-1:step]-x0)+ y0
     - y[:-1:step]))/atol
    return errscale**0.5 + 1
###═════════════════════════════════════════════════════════════════════

###═════════════════════════════════════════════════════════════════════
### COMPRESSION FUNCTIONS
def LSQ1(x,y,atol=1e-5, mins=10, verbosity=0, is_timed=False):
    '''Compresses the data of type y = f(x) using linear least squares fitting
    Works best if
    dy/dx < 0
    d2y/dx2 < 0
    '''
    if is_timed: t_start = time.perf_counter()
    zero = 1
    left = len(x)-1 - zero
    estimate = int(left/n_lines(x,y,x[0],y[0],int(left/5),atol) )+1
    #───────────────────────────────────────────────────────────────
    def initial_f2zero(n):
        '''Function such that n is optimal when initial_f2zero(n) = 0'''
        step = 1 if n<=mins*2 else int(n / ((n * 2 - mins)**0.5 + mins / 2))
        n += zero
        fit = lfit(x[:n+1:step], y[:n+1:step], 1)
        return max(abs(fit(x[0])- y[0]),abs(fit(x[n])- y[n])) - atol, fit
    #───────────────────────────────────────────────────────────────
    n2, fit = droot(initial_f2zero, -atol, estimate, left)
    zero += n2
    left -= n2
    x_c, y_c  = [0, x[zero-1]], [fit(0), fit(x[zero-1])]
    #───────────────────────────────────────────────────────────────
    def f2zero(n):
        '''Function such that n is optimal when f2zero(n) = 0'''
        step = 1 if n<=mins*2 else int(n/((n*2 - mins)**0.5 + mins/2))
        n += zero
        Dx = x[zero:n+1:step]-x_c[-1]
        Dy = y[zero:n+1:step]-y_c[-1]
        a = np.sum(Dy*Dx)/np.sum(Dx*Dx)
        b = y_c[-1] - a * x_c[-1]
        fit = lambda x: a*x + b
        return max(abs(fit(x[zero])-y[zero]),abs(fit(x[n])-y[n]))-atol, fit
    #───────────────────────────────────────────────────────────────
    while left > 0:
    
        estimate = int((n2 + left/(n_lines(x[zero+1:], y[zero+1:], 
                                           x_c[-1], y_c[-1], 
                                           int(left/5)+1, atol)))/2)
        estimate = min(left, estimate)
        n2, fit = droot(f2zero,-atol, estimate, left)
        
        zero += n2
        left -= n2
        x_c.append(x[zero-1])
        y_c.append(fit(x_c[-1]))

    if is_timed: t = time.perf_counter()-t_start
    if verbosity>0: 
        text = 'Length of compressed array\t%i'%len(x_c)
        text += '\nCompression factor\t%.3f %%' % (100*len(x_c)/len(x))
        if is_timed: text += '\nCompression time\t%.1f ms' % (t*1e3)
        print(text)
    print(x_c[-1])
    return np.array(x_c), np.array(y_c)
###═════════════════════════════════════════════════════════════════════
def LSQ10(x,y,atol=1e-5, mins=10, verbosity=0, is_timed=False):
    '''Compresses the data of type y = f(x) using linear least squares fitting
    Works best if
    dy/dx < 0
    d2y/dx2 < 0
    '''
    if is_timed: t_start = time.perf_counter()
    zero = 1
    left = len(x)-1 - zero
    estimate = int(left/n_lines(x,y,x[0],y[0],int(left/5),atol) )+1
    x_c, y_c  = [x[0]], [y[0]]
    #───────────────────────────────────────────────────────────────
    def f2zero(n):
        '''Function such that n is optimal when f2zero(n) = 0'''
        step = 1 if n<=mins*2 else int(n/((n*2 - mins)**0.5 + mins/2))
        n += zero
        Dx = x[zero:n+1:step]-x_c[-1]
        Dy = y[zero:n+1:step]-y_c[-1]
        a = np.sum(Dy*Dx)/np.sum(Dx*Dx)
        b = y_c[-1] - a * x_c[-1]
        fit = lambda x: a*x + b
        return max(abs(fit(x[zero])-y[zero]),abs(fit(x[n])-y[n]))-atol, fit
    #───────────────────────────────────────────────────────────────
    n2, fit = droot(f2zero,-atol, estimate, left)
        
    zero += n2
    left -= n2
    x_c.append(x[zero-1])
    y_c.append(fit(x_c[-1]))
    while left > 0:

        estimate = int((n2 + left/(n_lines(x[zero+1:], y[zero+1:], 
                                           x_c[-1], y_c[-1], 
                                           int(left/5)+1, atol)))/2)
        estimate = min(left, estimate)
        n2, fit = droot(f2zero,-atol, estimate, left)

        zero += n2
        left -= n2
        x_c.append(x[zero-1])
        y_c.append(fit(x_c[-1]))

    if is_timed: t = time.perf_counter()-t_start
    if verbosity>0: 
        text = 'Length of compressed array\t%i'%len(x_c)
        text += '\nCompression factor\t%.3f %%' % (100*len(x_c)/len(x))
        if is_timed: text += '\nCompression time\t%.1f ms' % (t*1e3)
        print(text)
    
    return np.array(x_c), np.array(y_c)
###═════════════════════════════════════════════════════════════════════
def pick(x,y,atol=1e-5, mins=30, verbosity=0, is_timed=False):

    if is_timed: t_start = time.perf_counter()

    x_slice = x
    y_slice = y

    limit = len(x)-3
    atol = atol
    #───────────────────────────────────────────────────────────────
    def f2zero(n,xs, ys,atol):
        n = int(n)+2
        step = 1 if n<=mins*2 else int(n/((n*2 - mins)**0.5 + mins/2))
        a = (ys[n] - ys[0])/(xs[n] - xs[0])
        b = ys[0] - a * xs[0]
        return max(np.abs(a*xs[0:n:step]+ b - ys[0:n:step]))-atol , None
    #───────────────────────────────────────────────────────────────
    x_c = [x_slice[0]]
    y_c = [y_slice[0]]
    step = int(limit/5)
    scaler = n_lines(x_slice,y_slice,x_slice[0],y_slice[0],int(limit/5),atol) 
    estimate = min(limit,int(limit/scaler+1))
    # print('estimate',estimate)

    n2, _ = droot(lambda n: f2zero(n, x_slice, y_slice, atol),
                    -atol, estimate, limit)
    # print('n2',n2)
    # print(estimate/n2-1)

    while n2-1 < limit:
        x_c.append(x_slice[n2])
        y_c.append(y_slice[n2])
        x_slice = x_slice[n2:]
        y_slice = y_slice[n2:]

        limit -= n2 + 1
        step = int(limit/5)
        errscale = 0.5*np.max(np.abs((y_slice[-1]- y_slice[0])
                            /(x_slice[-1] - x_slice[0])
                            * (x_slice[1:limit:step]-x_slice[0])
                            + y_slice[0] - y_slice[1:limit:step])) / atol
        scaler = errscale**0.5 + 1
        # print('from scaler',limit/scaler)
        estimate = min(limit,int((limit/scaler+n2)/2))
        # print('estimate',estimate)

        n2, _ = droot(lambda n: f2zero(n, x_slice, y_slice, atol),
                        -atol, estimate, limit)
        # print('n2',n2)
        # print(estimate/n2-1)
    #───────────────────────────────────────────────────────────────
    x_c = np.array(x_c.append(x_slice[-1]))
    y_c = np.array(y_c.append(y_slice[-1]))
    if is_timed: t = time.perf_counter()-t_start
    if verbosity>0: 
        text = 'Length of compressed array\t%i'%len(x_c)
        text += '\nCompression factor\t%.3f %%' % 100*len(x_c)/len(x)
        if is_timed: text += '\nCompression time\t%.1f ms' % (t*1e3)
        print(text)
    return x_c, y_c
###═════════════════════════════════════════════════════════════════════
def split(x,y,atol=1e-5, mins=100, verbosity=0, is_timed=False):
    t_start = time.perf_counter()
    def rec(a, b):
        n = b-a-1
        step = 1 if (n*2)<mins else int(round(n / ((n*2 - mins)**0.5 + mins/2)))

        x1, y1 = x[a], y[a]
        x2, y2 = x[b], y[b]
        err = lambda x, y: np.abs((y2- y1) /(x2 - x1)* (x - x1) + y1 - y)
        i = a + 1 + step*np.argmax(err(x[a+1:b-1:step], y[a+1:b-1:step]))
        return np.concatenate((rec(a, i), rec(i, b)[1:])) if err(x[i], y[i]) > atol else [a,b]
    indices= rec(0,len(x)-1)

    x_c = x[indices]
    y_c = y[indices]
    if is_timed: t = time.perf_counter()-t_start
    if verbosity>0:
        text = 'Length of compressed array\t%i'%len(x_c)
        text += '\nCompression factor\t%.3f %%' % 100*len(x_c)/len(x)
        if is_timed: text += '\nCompression time\t%.1f ms' % (t*1e3)
        print(text)
    return x_c, y_c

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
methods = {'LSQ1': LSQ1,
           'LSQ10': LSQ10,
           'pick': pick,
           'split': split}

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

def compress(*args, method='LSQ10', **kwargs):
    '''Wrapper for easier selection of compression method'''
    try:
        compressor = methods[method]
    except KeyError:
        raise NotImplementedError("Method not in the dictionary of methods")

    return compressor(*args,**kwargs)
