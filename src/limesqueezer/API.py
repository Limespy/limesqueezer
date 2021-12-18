#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import time
from numpy.polynomial import polynomial as poly
from collections import abc

#%%═════════════════════════════════════════════════════════════════════
# COMPRESSOR AUXILIARIES
#%%═════════════════════════════════════════════════════════════════════
## ERROR TERM
errorfunctions = {'maxmaxabs':
                  lambda r,t: np.amax(np.amax(np.abs(r), axis=0)/t-1),
                  'maxRMS':
                  lambda r,t: np.amax(np.sqrt(np.mean(r*r,axis=0))/t-1)}
#%%═════════════════════════════════════════════════════════════════════
## FITTING
def _fit_poly1(x_data, y_data, x0, y0):
    '''Does first degree polynomial fit and returns residuals and
     the y value at the end'''
    Dx = x_data - x0
    Dy = y_data - y0[0]
    a = np.matmul(Dx,Dy) / Dx.dot(Dx)
    return a*Dx.reshape([-1,1]) - Dy, a*Dx[-1]+y0[0], a
#───────────────────────────────────────────────────────────────────────
def _fit_poly2_2(x_data, y_data, x0, y0):
    '''Does second degree polynomial fit with two free parameters.
    Returns residuals and the y value at the end'''
    Dx = (x_data - x0)
    Dx2 = Dx*Dx
    print(f'shape of y_data\t{y_data.shape}')
    print(f'shape of Dx\t{Dx.shape}')
    Dy = y_data - y0[0]
    print(f'shape of Dy\t{Dy.shape}')
    yr = Dy/Dx.reshape([-1,1])
    x_mean, y_mean = np.mean(Dx), np.mean(yr, axis=0)
    a = np.matmul((Dx-x_mean),(yr-y_mean))/np.sum(Dx2)
    b = y_mean - a*x_mean
    print(f'shape of a\t{a.shape}')
    print(f'shape of b\t{b.shape}')
    y_next = [a*Dx2[-1]+ b*Dx[-1]+y0[0]]
    res = Dy - a*Dx2.reshape([-1,1]) - b*Dx.reshape([-1,1])
    print(f'shape of res\t{res.shape}')
    return res, y_next, (a,b)
#───────────────────────────────────────────────────────────────────────
def _fit_poly3(x_data, y_data, x0, y0):
    '''Does first degree polynomial fit and returns residuals and
     the y value at the end'''
    Dx = (x_data - x0)
    Dx2 = Dx*Dx
    Dx3 = Dx2 * Dx
    y_res = (y0[2]*Dx2 + y0[1]*Dx + y0[0]).reshape([-1,1])
    print(f'shape of y_data\t{y_data.shape}')
    print(f'shape of Dx\t{Dx.shape}')
    print(f'shape of y_res\t{y_res.shape}')
    
    Dy = y_data - y_res
    print(f'shape of Dy\t{Dy.shape}')
    print((y_data - y0[0]).shape)
    # hmm = np.matmul(Dx,y_data - y0[0]) / Dx.dot(Dx)
    # print(f'shape of hmm\t{hmm.shape}')
    a = np.matmul(Dx3,Dy) / Dx3.dot(Dx3)
    print(f'shape of a\t{a.shape}')
    y_next = [a*Dx3[-1]+y_res[-1], 3*a*Dx2[-1]+2*y0[2]*Dx[-1]+y0[1], 6*a*Dx[-1] + 2*y0[2]]
    res = a*Dx3.reshape([-1,1]) - Dy
    print(f'shape of res\t{res.shape}')
    return res, y_next, a
#───────────────────────────────────────────────────────────────────────
def _fit_poly4(x_data, y_data, x0, y0):
    '''Does first degree polynomial fit and returns residuals and
     the y value at the end'''
    Dx = (x_data - x0)
    Dx2 = Dx*Dx
    Dx3 = Dx2 * Dx
    y_res = (y0[2]*Dx2 + y0[1]*Dx + y0[0]).reshape([-1,1])
    print(f'shape of y_data\t{y_data.shape}')
    print(f'shape of Dx\t{Dx.shape}')
    print(f'shape of y_res\t{y_res.shape}')
    
    Dy = y_data - y_res
    print(f'shape of Dy\t{Dy.shape}')
    print((y_data - y0[0]).shape)
    # hmm = np.matmul(Dx,y_data - y0[0]) / Dx.dot(Dx)
    # print(f'shape of hmm\t{hmm.shape}')
    a = np.matmul(Dx3,Dy) / Dx3.dot(Dx3)
    print(f'shape of a\t{a.shape}')
    y_next = [a*Dx3[-1]+y_res[-1], 3*a*Dx2[-1]+2*y0[2]*Dx[-1]+y0[1], 6*a*Dx[-1] + 2*y0[2]]
    res = a*Dx3.reshape([-1,1]) - Dy
    print(f'shape of res\t{res.shape}')
    return res, y_next, a
#───────────────────────────────────────────────────────────────────────
fitfunctions = {'poly1': _fit_poly1,
                'poly2_2': _fit_poly2_2,
                'poly3': _fit_poly3,
                'poly4': _fit_poly4}
#%%═════════════════════════════════════════════════════════════════════
## ROOT FINDING
def interval(f,x1,y1,x2,y2,fit1):
    '''Returns the last x where f(x)<0'''
    while x2 - x1 > 2:
        # Arithmetic mean between linear estimate and half
        x = int((x1-y1/(y2-y1)*(x2-x1) + (x2 + x1)/2)/2) + 1
        if x == x1:    # To stop repetition in close cases
            x += 1
        elif x == x2:
            x -= 1

        y, fit = f(x)
        if y > 0:
            x2, y2 = x, y
        else: 
            x1, y1, fit1 = x, y, fit

    if x2 - x1 == 2: # Points have only on e point in between
        y, fit = f(x1+1) # Testing that point
        return (x1+1, fit) if (y <0) else (x1, fit1) # If under, give that fit
    else:
        return x1, fit1
#───────────────────────────────────────────────────────────────────────
def droot(f, y0, x2, limit):
    '''Finds the upper limit to interval
    '''
    x1, y1 = 0, y0
    y2, fit2 = f(x2)
    fit1 = None
    while y2 < 0:
        x1, y1, fit1 = x2, y2, fit2
        x2 *= 2
        if x2 >= limit:
            y2, fit2 = f(limit)
            if y2<0:
                return limit, fit2
            else:
                x2 = limit
                break
        y2, fit2 = f(x2)
    return interval(f,x1, y1, x2, y2,fit1)

#───────────────────────────────────────────────────────────────────────
# @numba.jit(nopython=True,cache=True)
def n_lines(x,y,x0,y0,ytol):
    '''Estimates number of lines required to fit within error tolerance'''
    if (length := len(x)) > 1:
        inds = np.rint(np.linspace(1,length-2,int(length**0.5))).astype(int)
        return (0.5*np.amax(np.abs(
                                    (y[-1]-y0)/(x[-1]-x0)*(x[inds]-x0).reshape([-1,1])
                                    - (y[inds]-y0)
                                    ) / ytol,
                                     axis=0)
                                     )**0.5 + 1
    else:
        return 1
    
###═════════════════════════════════════════════════════════════════════
### BLOCK COMPRESSION
def LSQ1(x,y,ytol=1e-2, mins=10, verbosity=0, is_timed=False):
    '''Compresses the data of type y = f(x) using linear least squares fitting
    Works best if
    dy/dx < 0
    d2y/dx2 < 0
    '''
    if is_timed: t_start = time.perf_counter()
    zero = 1
    end = len(x)-1 - zero
    estimate = int(end / n_lines(x,y,x[0],y[0],int(end/5),ytol) )
    #───────────────────────────────────────────────────────────────
    def initial_f2zero(n):
        '''Function such that n is optimal when initial_f2zero(n) = 0'''
        step = 1 if n<=mins*2 else int(n / ((n * 2 - mins)**0.5 + mins / 2))
        n += zero
        fit = poly.Polynomial.fit(x[:n+1:step], y[:n+1:step], 1)
        return max(abs(fit(x[0])- y[0]), abs(fit(x[n])- y[n])) - ytol, fit
    #───────────────────────────────────────────────────────────────
    end, fit = droot(initial_f2zero, -ytol, estimate, end)
    zero += end
    end -= end
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
        return max(abs(fit(x[zero])-y[zero]),abs(fit(x[n])-y[n]))-ytol, fit
    #───────────────────────────────────────────────────────────────
    while end > 0:
    
        estimate = int((end + end/(n_lines(x[zero+1:], y[zero+1:], 
                                           x_c[-1], y_c[-1], ytol)))/2)
        estimate = min(end, estimate)
        end, fit = droot(f2zero,-ytol, estimate, end)
        
        zero += end
        end -= end
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
def LSQ10(x, y, ytol=1e-2, verbosity=0, is_timed=False):
    '''Compresses the data of 1-dimensional system of equations
    i.e. single input variable and one or more output variable
    '''
    runtime = None
    if is_timed: t_start = time.perf_counter()
    start = 1 # Index of starting point for looking for optimum
    end = len(x) - 2 # Number of uncompressed datapoints -2, i.e. the index
    offset = -1
    fit = None
    ytol = np.array(ytol)
    x_c, y_c = [], []
    if len(y.shape) == 1: # Converting to correct shape for this function
        y = y.reshape(len(x),1)
    elif y.shape[0] != len(x):
        y = y.T
    #───────────────────────────────────────────────────────────────
    def _f2zero(n):
        '''Function such that n is optimal when f2zero(n) = 0'''
        inds = np.linspace(start, n + start, int((n+1)**0.5)+ 2).astype(int)

        Dx = x[inds] - x_c[-1]
        Dy = y[inds] - y_c[-1]

        a = np.matmul(Dx,Dy) / Dx.dot(Dx)
        b = y_c[-1] - a * x_c[-1]

        errmax = np.amax(np.abs(a*x[inds].reshape([-1,1]) + b - y[inds]),
                         axis=0)

        return np.amax(errmax/ytol-1), (a,b)
    #───────────────────────────────────────────────────────────────
    while end > 0:
        x_c.append(x[offset + start])
        y_c.append(fit[0]*x_c[-1] + fit[1] if fit else y[offset + start])
        start += offset + 1 # Start shifted by the number compressed
        # Estimated number of lines needed
        lines = n_lines(x[start:], y[start:], x_c[-1], y_c[-1], ytol)
        # Arithmetic mean between previous step length and line estimate,
        # limited to end index of the array
        estimate = min(end, np.amin(((offset + (end+1) / lines)/2)).astype(int))

        offset, fit = droot(_f2zero, -1, estimate, end)
        end -= offset + 1
    # Last data point is same as in the uncompressed data
    x_c.append(x[-1])
    y_c.append(y[-1])

    if is_timed: runtime = time.perf_counter() - t_start
    
    return np.array(x_c).reshape(-1,1), np.array(y_c), runtime
###═════════════════════════════════════════════════════════════════════
def pick(x,y,ytol=1e-2, mins=30, verbosity=0, is_timed=False):
    '''Returns inds of data points to select'''

    if is_timed: t_start = time.perf_counter()

    zero = 1
    end = len(x)- 1 - zero
    estimate = int(end/n_lines(x,y,x[0],y[0],ytol) )+1
    inds = [0]
    #───────────────────────────────────────────────────────────────────
    def f2zero(n,xs, ys,ytol):
        n_steps = n+1 if n+1<=mins else int((n+1 - mins)**0.5 + mins)
        inds_test = np.rint(np.linspace(zero,n+ zero,n_steps)).astype(int)

        a = (y[n+zero] - y[zero])/(x[n+zero] - x[zero])
        b = y[zero] - a * xs[zero]

        errmax = np.amax(np.abs(a*x[inds_test].reshape([-1,1]) + b - y[inds_test]),axis=0)
        return np.amax(errmax/ytol-1), None
    #───────────────────────────────────────────────────────────────────
    while end > 0:
        estimate = int((end + end/(n_lines(x[zero:], y[zero:], 
                                           x[inds[-1]], y[inds[-1]], ytol)))/2)
        estimate = min(end, estimate)
        end, _ = droot(f2zero,-ytol, estimate, end)
        end += 1
        zero += end
        end -= end
        
        inds.append(zero-1)

    if is_timed: t = time.perf_counter()-t_start
    if verbosity>0: 
        text = 'Length of compressed array\t%i'%len(inds)
        text += '\nCompression factor\t%.3f %%' % (100*len(inds)/len(x))
        if is_timed: text += '\nCompression time\t%.1f ms' % (t*1e3)
        print(text)
    
    return np.array(inds)
###═════════════════════════════════════════════════════════════════════
def split(x,y,ytol=1e-2, mins=100, verbosity=0, is_timed=False):
    t_start = time.perf_counter()
    def rec(a, b):
        n = b-a-1
        step = 1 if (n*2)<mins else int(round(n / ((n*2 - mins)**0.5 + mins/2)))

        x1, y1 = x[a], y[a]
        x2, y2 = x[b], y[b]
        err = lambda x, y: np.abs((y2- y1) /(x2 - x1)* (x - x1) + y1 - y)
        i = a + 1 + step*np.argmax(err(x[a+1:b-1:step], y[a+1:b-1:step]))
        return np.concatenate((rec(a, i), rec(i, b)[1:])) if err(x[i], y[i]) > ytol else [a,b]
    inds = rec(0,len(x)-1)

    if is_timed: t = time.perf_counter()-t_start
    if verbosity>0:
        text = 'Length of compressed array\t%i'%len(inds)
        text += '\nCompression factor\t%.3f %%' % 100*len(inds)/len(x)
        if is_timed: text += '\nCompression time\t%.1f ms' % (t*1e3)
        print(text)
    return inds
###═════════════════════════════════════════════════════════════════════
### STREAM COMPRESSION
class _StreamCompressedContainer(abc.Sized):
    '''Class for doing stream compression for data of 1-dimensional
    system of equations 
    i.e. single input variable and one or more output variable
    '''
    def __init__(self, x0 ,y0, mins=20, ytol=1e-2):
        self.xb = []
        self.yb = [] # Variables are columns, e.g. 3xn
        self._x = [x0]
        self._y = [np.array(y0)]
        self.start = 0 # Index of starting point for looking for optimum
        self.end = 2 # Index of end point for looking for optimum
        self.mins = mins
        self.ytol = np.array(ytol)
        self.state = 'open'
    #───────────────────────────────────────────────────────────────────
    @property # An optimization
    def x(self):
        return self._x if self.state == 'closed' else np.array(self._x + self.xb[-1:])
    #───────────────────────────────────────────────────────────────────
    @property # An optimization
    def y(self):
        return self._y if self.state == 'closed' else np.array(self._y + self.yb[-1:])
    #───────────────────────────────────────────────────────────────────
    def _f2zero(self,n):
        '''Function such that n is optimal when f2zero(n) = 0'''
        #inds = np.rint(np.linspace(0,n,n_steps)).astype(int)
        inds = np.linspace(0, n, int((n+1)**0.5)+ 2).astype(int)

        Dx = self.xb[inds] - self._x[-1]
        Dy = self.yb[inds] - self._y[-1]
        
        a = np.matmul(Dx,Dy) / Dx.dot(Dx)
        b = self._y[-1] - a * self._x[-1]
        errmax = np.amax(np.abs(a*self.xb[inds].reshape([-1,1]) + b - self.yb[inds]),axis=0)
        return np.amax(errmax/self.ytol-1), (a,b)
    #───────────────────────────────────────────────────────────────────
    def compress(self):
        offset, fit = interval(self._f2zero,self.start,self.tol1, self.limit,self.tol2,self.fit1)
        self.start, self.end, self.tol1 = 0, offset, -1.
        self._x.append(self.xb[offset])
        self._y.append(fit[0]*self._x[-1] + fit[1])

        self.xb = self.xb[offset:]
        self.yb = self.yb[offset:]
    #───────────────────────────────────────────────────────────────────
    def __call__(self,x_input,y_input):
        self.xb.append(x_input)
        self.yb.append(y_input)
        self.limit = len(self.xb) - 1
        if  self.limit > self.end:
            self.xb, self.yb = np.array(self.xb), np.array(self.yb)
            
            self.tol2, self.fit2 = self._f2zero(self.limit)
            if self.tol2 < 0:
                self.start, self.tol1, self.fit1 = self.end, self.tol2, self.fit2
                self.end *= 2
            else:
                self.compress()
            self.xb, self.yb = list(self.xb), list(self.yb)
        return len(self._x), len(self.xb)
    #───────────────────────────────────────────────────────────────────
    def close(self):
        self.state = 'closing'
        self.xb, self.yb = np.array(self.xb), np.array(self.yb)
        self.start, self.limit  = 0, len(self.xb) - 1
        self.tol2, self.fit2 = self._f2zero(self.limit)

        while self.tol2 > 0:
            self.compress()
            self.limit  = len(self.xb) - 1
            self.tol1, self.fit1 = -1, self.fit2
            self.tol2, self.fit2 = self._f2zero(self.limit)
        
        self._x.append(self.xb[-1])
        self._y.append(self.yb[-1])
        self._x, self._y = np.array(self._x), np.array(self._y)
        self.state = 'closed'
    #───────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self._x) +len(self.xb)
    #───────────────────────────────────────────────────────────────────
    def __str__(self):
        s = 'x = ' + str(self.x)
        s += ' y = ' + str(self.y)
        s += ' ytol = ' + str(self.ytol)
        return s
    #───────────────────────────────────────────────────────────────────
###═════════════════════════════════════════════════════════════════════
class Stream():
    '''Context manager for stream compression of data of 
    1 dimensional system of equations'''
    def __init__(self, x0 ,y0, mins=20, ytol=1e-2):
        self.x0 = x0
        self.y0 = y0 # Variables are columns, e.g. 3xn
        self.mins = mins
        self.ytol = ytol # Y value tokerances
    #───────────────────────────────────────────────────────────────────
    def __enter__(self):
        self.container = _StreamCompressedContainer(self.x0, self.y0, 
                                             mins=self.mins, ytol=self.ytol)
        return self.container
    #───────────────────────────────────────────────────────────────────
    def __exit__(self, exc_type, exc_value, traceback):
        self.container.close()
#%%═════════════════════════════════════════════════════════════════════
# WRAPPING

methods = {'LSQ1': LSQ1,
           'LSQ10': LSQ10,
           'pick': pick,
           'split': split}

def compress(*args, method='LSQ10', **kwargs):
    '''Wrapper for easier selection of compression method'''
    try:
        compressor = methods[method]
    except KeyError:
        raise NotImplementedError("Method not in the dictionary of methods")

    return compressor(*args,**kwargs)

