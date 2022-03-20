#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import numba
import time

from collections import abc
import matplotlib.pyplot as plt

global G
G = {}
G['timed'] = False
G['debug'] = False
#%%═════════════════════════════════════════════════════════════════════
# COMPRESSOR AUXILIARIES
# ~ sqrt(n + 2) equally spaced integers including the i
# sqrtrange = lambda i: np.linspace(0, i, round((i+2)**0.5)).astype(int)
def sqrtrange(i: int):
    '''~ sqrt(n + 2) equally spaced integers including the i'''
    inds = np.arange(0, i + 1, round(i**0.5))
    inds[-1] = i
    return inds
#%%═════════════════════════════════════════════════════════════════════
## ERROR TERM
# @numba.jit(nopython=True, cache=True)
def _maxmaxabs(r: np.ndarray, t: np.ndarray) -> float:
    return max(np.amax(np.abs(r, out = r), axis = 0) - t)

def _maxRMS(r: np.ndarray,t: np.ndarray)-> float:
    return np.amax(np.sqrt(np.mean(r * r, axis = 0)) - t)

def _maxsumabs(r: np.ndarray,t: np.ndarray) -> float:
    return np.amax(np.sum(np.abs(r) - t, axis = 0))

errorfunctions = {'maxmaxabs': _maxmaxabs,
                  'maxRMS':_maxRMS}
#%%═════════════════════════════════════════════════════════════════════
## FITTING
#%%═════════════════════════════════════════════════════════════════════
## ROOT FINDING
def interval(f,x1,y1,x2,y2,fit1):
    '''Returns the last x where f(x)<0'''
    is_debug = G['debug']
    if is_debug:
        G['mid'], = G['ax_root'].plot(x1, y1,'.', color='blue')
    while x2 - x1 > 2:
        if is_debug:
            input('Calculating new attempt in interval\n')
        # Arithmetic mean between linear estimate and half
        x_mid = int((x1 - y1 / (y2 - y1) * (x2 - x1) + (x2 + x1) / 2) / 2) + 1
        # x_mid = int((x2 + x1) / 2)
        if x_mid == x1:    # To stop repetition in close cases
            x_mid += 1
        elif x_mid == x2:
            x_mid -= 1

        y_mid, fit = f(x_mid)
        if is_debug:
            print(f'{x_mid=}')
            print(f'{y_mid=}')
            G['mid'].set_xdata(x_mid)
            G['mid'].set_ydata(y_mid)
        if y_mid > 0:
            if is_debug:
                input('Error over tolerance\n')
                G['ax_root'].plot(x2, y2,'.', color='black')
            x2, y2 = x_mid, y_mid
            if is_debug:
                G['xy2'].set_xdata(x2)
                G['xy2'].set_ydata(y2)
        else:
            if is_debug:
                input('Error under tolerance\n')
                G['ax_root'].plot(x1, y1,'.', color='black')
            x1, y1, fit1 = x_mid, y_mid, fit
            if is_debug:
                G['xy1'].set_xdata(x1)
                G['xy1'].set_ydata(y1)

    if x2 - x1 == 2: # Points have only on e point in between
        if is_debug:
            input('Points have only one point in between\n')
        y_mid, fit = f(x1+1) # Testing that point
        return (x1+1, fit) if (y_mid <0) else (x1, fit1) # If under, give that fit
    else:
        if is_debug:
            input('Points have no point in between\n')
        return x1, fit1
#───────────────────────────────────────────────────────────────────────
def droot(f, y1, x2, limit):
    '''Finds the upper limit to interval
    '''
    is_debug = G['debug']
    x1 = 0
    y2, fit2 = f(x2)
    if is_debug:
        G['xy1'], = G['ax_root'].plot(x1, y1,'.', color='green')
        G['xy2'], = G['ax_root'].plot(x2, y2,'.', color='blue')
    fit1 = None
    while y2 < 0:
        if is_debug:
            input('Calculating new attempt in droot\n')
            G['ax_root'].plot(x1, y1,'.', color='black')
        x1, y1, fit1 = x2, y2, fit2
        x2 *= 2
        x2 += 1
        if is_debug:
            print(f'{limit=}')
            print(f'{x1=}')
            print(f'{y1=}')
            print(f'{x2=}')
            G['xy1'].set_xdata(x1)
            G['xy1'].set_ydata(y1)
            G['xy2'].set_xdata(x2)
        if x2 >= limit:
            if is_debug:
                G['ax_root'].plot([limit, limit], [y1,0],'.', color='blue')
            y2, fit2 = f(limit)
            if y2<0:
                if is_debug:
                    input('End reached within tolerance\n')
                return limit, fit2
            else:
                if is_debug:
                    input('End reached outside tolerance\n')
                x2 = limit
                break
        y2, fit2 = f(x2)
        if is_debug:
            print(f'{y2=}')
            G['xy2'].set_ydata(y2)
    if is_debug:
        G['xy2'].set_color('red')
        input('Points for interval found\n')
    return interval(f, x1, y1, x2, y2, fit1)
#───────────────────────────────────────────────────────────────────────
# @numba.jit(nopython=True,cache=True)
def n_lines(x: np.ndarray, y: np.ndarray, x0: float, y0: np.ndarray, tol: float
            ) -> float:
    '''Estimates number of lines required to fit within error tolerance'''

    if (length := len(x)) > 1:
        inds = sqrtrange(length - 2) # indices so that x[-1] is not included
        res = (y[-1] - y0) / (x[-1] - x0)*(x[inds] - x0).reshape([-1,1]) - (y[inds] - y0)
        # print(f'sqrtmaxmaxabs {_maxmaxabs(res, tol)** 0.5}')
        # print(f'sqrtmaxsumabs {_maxsumabs(res, tol)** 0.5}')
        # print(f'sqrtmaxRMS {_maxRMS(res, tol)** 0.5}')
        return (0.5 * _maxRMS(res, tol) + 1) ** 0.5 + 1
    else:
        return 1.

###═════════════════════════════════════════════════════════════════════
### BLOCK COMPRESSION
def LSQ10(x: np.ndarray, y: np.ndarray, tol = 1e-2, errorfunction = 'maxmaxabs'
          ) -> tuple:
    '''Compresses the data of 1-dimensional system of equations
    i.e. single input variable and one or more output variable
    '''
    is_debug = G['debug']
    if G['timed']:
        G['t_start'] = time.perf_counter()
    if is_debug:
        G['x'], G['y'] = x, y
        G['fig'], axs = plt.subplots(3,1)
        for ax in axs:
            ax.grid()
        G['ax_data'], G['ax_res'], G['ax_root'] = axs

        G['ax_data'].fill_between(x, y - tol, y + tol, alpha=.3, color='blue')

        G['line_fit'], = G['ax_data'].plot(0,0,'-',color='orange')

        G['ax_root'].set_ylabel('Tolerance left')

        plt.ion()
        plt.show()

    start = 1 # Index of starting point for looking for optimum
    end = len(x) - 1 # Number of uncompressed datapoints -1, i.e. the last index
    offset = 0
    fit = None
    tol = np.array(tol)
    x = x.reshape([-1, 1])

    errf = errorfunctions[errorfunction]
    fitset = Poly1
    f_fit = fitset.fit
    f_y = fitset.y_from_fit

    x_c, y_c = [x[0]], [np.array(y[0])]
    #───────────────────────────────────────────────────────────────
    def _f2zero(i: int) -> tuple:
        '''Function such that i is optimal when f2zero(i) = 0'''
        inds = sqrtrange(i) + start
        residuals, fit = f_fit(x[inds], y[inds], x_c[-1], y_c[-1])
        if is_debug:
            indices_all = np.arange(-1, i + 1) + start 
            G['x_plot'] = G['x'][indices_all]
            G['y_plot'] = Poly1.y_from_fit(fit, G['x_plot'])
            G['line_fit'].set_xdata(G['x_plot'])
            G['line_fit'].set_ydata(G['y_plot'])
            res_all = G['y_plot'] - G['y'][indices_all].reshape(-1,1)
            G['ax_res'].clear()
            G['ax_res'].grid()
            G['ax_res'].set_ylabel('Residual relative to tolerance')
            G['ax_res'].plot(indices_all - start, np.abs(res_all) / tol -1,
                             '.', color = 'blue', label='ignored')
            G['ax_res'].plot(inds - start, np.abs(residuals) / tol-1,
                             'o', color='red', label='sampled')
            G['ax_res'].legend()
            input('Fitting\n')
        return errf(residuals, tol), fit
    #───────────────────────────────────────────────────────────────
    limit = end - start
    # Estimation for the first offset
    offset = min(limit,
                 round((limit + 1) / n_lines(x[start:round(end / 2)],
                                             y[start:round(end / 2)],
                                             x_c[-1], y_c[-1], tol)))
    if is_debug:
        input('Starting\n')
        print(f'{offset=}')
    for _ in range(end): # Prevents infinite loop in case error
        
        offset, fit = droot(_f2zero, -1, offset, limit)

        if fit is None: raise RuntimeError('Fit not found')
        if is_debug:
            print(f'err {errf(f_y(fit, x[start + offset]) - y[start + offset], tol)}')
            print(f'{start=}')
            print(f'{offset=}')
            print(f'{end=}')
            print(f'{fit=}')
            G['ax_root'].clear()
            G['ax_root'].grid()
            G['ax_root'].set_ylabel('Maximum residual')
            G['ax_data'].plot(G['x_plot'], G['y_plot'], color='red')
        
        start += offset + 1 # Start shifted by the number compressed and the
        if start > end:
            break
        x_c.append(x[start - 1])
        y_c.append(f_y(fit, x_c[-1]).flatten())
        limit = end - start
        offset = min(limit, offset) # Setting up to be next estimation

        if is_debug:
            G['ax_data'].plot(x_c[-1], y_c[-1],'.',color='green')
            input('Next iteration\n')
    else:
        raise StopIteration('Maximum number of iterations reached')
    # Last data point is same as in the uncompressed data
    x_c.append(x[-1])
    y_c.append(y[-1])

    if G['timed']:
        G['runtime'] = time.perf_counter() - G['t_start']
    
    if is_debug:
        plt.ioff()
    return np.array(x_c).reshape(-1,1), np.array(y_c)
###═════════════════════════════════════════════════════════════════════
def pick(x, y, tol=1e-2, mins=30, verbosity=0, is_timed=False):
    '''Returns inds of data points to select.
    Should be faster than LSQ based'''

    if is_timed: t_start = time.perf_counter()

    zero = 1
    end = len(x)- 1 - zero
    estimate = int(end/n_lines(x,y,x[0],y[0],tol) )+1
    inds = [0]
    #───────────────────────────────────────────────────────────────────
    def f2zero(n,xs, ys, tol):
        n_steps = n+1 if n+1<=mins else int((n+1 - mins)**0.5 + mins)
        inds_test = np.rint(np.linspace(zero,n+ zero,n_steps)).astype(int)

        a = (y[n+zero] - y[zero])/(x[n+zero] - x[zero])
        b = y[zero] - a * xs[zero]

        errmax = np.amax(np.abs(a*x[inds_test].reshape([-1,1]) + b - y[inds_test]),axis=0)
        return np.amax(errmax/tol-1), None
    #───────────────────────────────────────────────────────────────────
    while end > 0:
        estimate = int((end + end/(n_lines(x[zero:], y[zero:], 
                                           x[inds[-1]], y[inds[-1]], tol)))/2)
        estimate = min(end, estimate)
        end, _ = droot(f2zero,-tol, estimate, end)
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
def split(x, y, tol=1e-2, mins=100, verbosity=0, is_timed=False):
    '''Returns inds of data points to select.
    Should be faster pick'''
    t_start = time.perf_counter()
    def rec(a, b):
        n = b-a-1
        step = 1 if (n*2)<mins else int(round(n / ((n*2 - mins)**0.5 + mins/2)))

        x1, y1 = x[a], y[a]
        x2, y2 = x[b], y[b]
        err = lambda x, y: np.abs((y2- y1) /(x2 - x1)* (x - x1) + y1 - y)
        i = a + 1 + step*np.argmax(err(x[a+1:b-1:step], y[a+1:b-1:step]))
        return np.concatenate((rec(a, i), rec(i, b)[1:])) if err(x[i], y[i]) > tol else [a,b]
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
    def __init__(self, x0 ,y0, mins=20, tol=1e-2):
        self.xb = []
        self.yb = [] # Variables are columns, e.g. 3xn
        self._x = [x0]
        self._y = [np.array(y0)]
        self.start = 0 # Index of starting point for looking for optimum
        self.end = 2 # Index of end point for looking for optimum
        self.mins = mins
        self.tol = np.array(tol)
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
        return np.amax(errmax/self.tol-1), (a,b)
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
        s += ' tol = ' + str(self.tol)
        return s
    #───────────────────────────────────────────────────────────────────
###═════════════════════════════════════════════════════════════════════
class Stream():
    '''Context manager for stream compression of data of 
    1 dimensional system of equations'''
    def __init__(self, x0 ,y0, mins=20, tol=1e-2):
        self.x0 = x0
        self.y0 = y0 # Variables are columns, e.g. 3xn
        self.mins = mins
        self.tol = tol # Y value tokerances
    #───────────────────────────────────────────────────────────────────
    def __enter__(self):
        self.container = _StreamCompressedContainer(self.x0, self.y0, 
                                             mins=self.mins, tol=self.tol)
        return self.container
    #───────────────────────────────────────────────────────────────────
    def __exit__(self, exc_type, exc_value, traceback):
        self.container.close()
#%%═════════════════════════════════════════════════════════════════════
# WRAPPING

methods = {'LSQ10': LSQ10,
           'pick': pick,
           'split': split}

def compress(*args, method='LSQ10', **kwargs):
    '''Wrapper for easier selection of compression method'''
    try:
        compressor = methods[method]
    except KeyError:
        raise NotImplementedError("Method not in the dictionary of methods")

    return compressor(*args,**kwargs)

#%%═════════════════════════════════════════════════════════════════════
# CUSTOM FUNCTIONS

class Poly1:
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    # @numba.jit(nopython=True, cache=True)
    def fit(x: np.ndarray, y: np.ndarray, x0, y0: np.ndarray) -> tuple:
        '''Takes block of data, previous fitting parameters and calculates next fitting parameters'''

        Dx = x - x0
        Dy = y - y0
        a = Dx.T @ Dy / Dx.T.dot(Dx)
        b = y0 - a * x0
        return (a * Dx - Dy, (a,  b))
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    def y_from_fit(fit: tuple, x: np.ndarray) -> np.ndarray:
        '''Converts the fitting parameters and x to storable y values'''
        return fit[0]*x.reshape([-1,1]) + fit[1]
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    def fit_from_end(ends: np.ndarray) -> tuple:
        '''Takes storable y values and return fitting parameters'''
        a = (ends[1,1:]- ends[0,1:]) / (ends[1,0] - ends[0,0])
        b = ends[0,1:] - a * ends[0,0]
        return a, b
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    def full_reconstruct(fit_array: np.ndarray, x_values: np.ndarray):
        '''Takes array of fitting parameters and constructs whole function'''
        raise NotImplementedError

