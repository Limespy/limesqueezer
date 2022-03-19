#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import time
from numpy.polynomial import polynomial as poly
from collections import abc
import matplotlib.pyplot as plt

global G
G = {}
G['timed'] = False
G['debug'] = False
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
        x_mid = int((x1-y1/(y2-y1)*(x2-x1) + (x2 + x1)/2)/2) + 1

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
def droot(f, y0, x2, limit):
    '''Finds the upper limit to interval
    '''
    is_debug = G['debug']
    x1, y1 = 0, y0
    
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
def LSQ10(x, y, ytol=1e-2, errorfunction = 'maxmaxabs'):
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

        
        G['ax_data'].fill_between(x, y - ytol, y + ytol, alpha=.3, color='blue')

        G['line_fit'], = G['ax_data'].plot(0,0,'-',color='orange')

        G['ax_root'].set_ylabel('Tolerance left')

        plt.ion()
        plt.show()

    start = 1 # Index of starting point for looking for optimum
    end = len(x) - 1 # Number of uncompressed datapoints -1, i.e. the last index
    offset = 0
    fit = None
    ytol = np.array(ytol)
    

    errf = errorfunctions[errorfunction]
    fitset = Poly1
    f_fit = fitset.fit
    f_y = fitset.y_from_fit

    y = y.reshape(-1,1)
    x_c, y_c = [x[0]], [np.array(y[0])]
    #───────────────────────────────────────────────────────────────
    def _f2zero(n):
        '''Function such that n is optimal when f2zero(n) = 0'''
        inds = np.linspace(start, n + start, int((n+1)**0.5)+ 2).astype(int)
        residuals, fit = f_fit(x[inds], y[inds], x_c[-1], y_c[-1])
        if is_debug:
            indices_all = np.arange(start, start + int(n) + 1)
            G['x_plot'] = G['x'][indices_all]
            G['y_plot'] = Poly1.y_from_fit(fit, G['x_plot'])
            G['line_fit'].set_xdata(G['x_plot'])
            G['line_fit'].set_ydata(G['y_plot'])
            res_all = G['y_plot'] - G['y'][indices_all].reshape(-1,1)
            G['ax_res'].clear()
            G['ax_res'].grid()
            G['ax_res'].set_ylabel('Residual relative to tolerance')
            G['ax_res'].plot(indices_all - start, np.abs(res_all) / ytol -1,
                             '.', color = 'blue', label='ignored')
            G['ax_res'].plot(inds - start, np.abs(residuals) / ytol-1,
                             'o', color='red', label='sampled')
            G['ax_res'].legend()
            input('Fitting\n')
        return errf(residuals, ytol), fit
    #───────────────────────────────────────────────────────────────

    for _ in range(end): # Prevents infinite loop in case error
        if is_debug:
                input('Next iteration\n')
        # Estimated number of lines needed
        lines = n_lines(x[start:], y[start:], x_c[-1], y_c[-1], ytol)
        # Arithmetic mean between previous step length and line estimate,
        # limited to end index of the array
        limit = end - start
        estimate = min(limit, np.amin(((offset + (limit+1) / lines)/2)).astype(int))
        if is_debug:
            print(f'{lines=}')
            print(f'{estimate=}')
        offset, fit = droot(_f2zero, -1, estimate, limit)
        if is_debug:
            print(f'err {errf(f_y(fit, x[start + offset]) - y[start + offset], ytol)}')
            print(f'{start=}')
            print(f'{offset=}')
            print(f'{end=}')
            print(f'{fit=}')
            G['ax_root'].clear()
            G['ax_root'].grid()
            G['ax_root'].set_ylabel('Maximum residual')
            G['ax_data'].plot(G['x_plot'], G['y_plot'], color='red')
        start += offset + 1 # Start shifted by the number compressed and the
        if end <=  start:
            break
        x_c.append(x[start - 1])
        y_c.append(f_y(fit, x_c[-1]).flatten())
    else:
        raise Warning('Maximum number of iterations reached')
    # Last data point is same as in the uncompressed data
    x_c.append(x[-1])
    y_c.append(y[-1])

    if G['timed']:
        G['runtime'] = time.perf_counter() - G['t_start']
    
    if is_debug:
        plt.ioff()
    return np.array(x_c).reshape(-1,1), np.array(y_c)
###═════════════════════════════════════════════════════════════════════
def pick(x,y,ytol=1e-2, mins=30, verbosity=0, is_timed=False):
    '''Returns inds of data points to select'''

    if is_timed: t_start = time.perf_counter()

    zero = 1
    end = len(x)- 1 - zero
    estimate = int(end/n_lines(x,y,x[0],y[0],ytol) )+1
    inds = [0]
    #───────────────────────────────────────────────────────────────────
    def f2zero(n,xs, ys, ytol):
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
    def fit(x: np.ndarray, y: np.ndarray, x0, y0: np.ndarray) -> tuple:
        '''Takes block of data, previous fitting parameters and calculates next fitting parameters'''
        Dx = x - x0
        Dy = y - y0.reshape([1, -1])

        a = np.matmul(Dx,Dy) / Dx.dot(Dx)
        b = y0 - a * x0
        return (a*x.reshape([-1, 1]) + b - y, (a,  b))
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

