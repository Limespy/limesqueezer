#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
API
========================================================================

Connection point for all package utilities to be provided
'''

import collections

try:
    import numba
except ModuleNotFoundError:
    pass

import numpy as np
import sys
import time
import types
from bisect import bisect_left
import matplotlib.pyplot as plt

from . import GLOBALS
from . import reference as ref # Careful with this circular import

# This global dictionary _G is for passing some telemtery and debug arguments
global _G
_G = {}
_G['timed'] = False
_G['debug'] = False
_G['profiling'] = False

import math
#%%═════════════════════════════════════════════════════════════════════
# AUXILIARIES
def to_ndarray(item, shape = ()) :
    if not hasattr(item, '__iter__'): # Not some iterable
        if -1 in shape: # Array of shape length of dimensions with one item
            return np.array(item, ndmin = len(shape))
        else:
            return np.full(shape, item) # Array of copies in the shape 
    elif not isinstance(item, np.ndarray): # Iterable into array
        item = np.array(item)
    return item if shape == () else item.reshape(shape)
#───────────────────────────────────────────────────────────────────────
def sqrtrange_python(start: int, n: int):
    '''~ sqrt(n + 2) equally spaced integers including the n'''
    inds = np.arange(start, n + start + 1, round((n + 1) ** 0.5) )
    inds[-1] = n + start
    return inds
#───────────────────────────────────────────────────────────────────────
@numba.jit(nopython = True, cache = True)
def sqrtrange_numba(start: int, n: int):
    '''~ sqrt(n + 2) equally spaced integers including the n'''
    inds = np.arange(start, n + start + 1, round(math.sqrt(n + 1)) )
    inds[-1] = n + start
    return inds
#───────────────────────────────────────────────────────────────────────
_sqrtrange = (sqrtrange_python, sqrtrange_numba)
#───────────────────────────────────────────────────────────────────────
def wait(text = ''):
    if input(text) in ('e', 'q', 'exit', 'quit'): sys.exit()
#%%═════════════════════════════════════════════════════════════════════
## ERROR TERM
def _maxmaxabs_python(r: np.ndarray, t: np.ndarray) -> float:
    '''Python version'''
    r = np.abs(r)
    # print(f'{r.shape=}')
    # print(f'{t.shape=}')
    r_max = r[0,0]

    for r0 in r[:,0]:
        if r0 > r_max: r_max = r0

    m = r_max - t[0]

    for i, k in enumerate(t[1:]):
        for r0 in r[:,i]:
            if r0 > r_max: r_max = r0
        m = max(m, r_max - k)
    return m
#───────────────────────────────────────────────────────────────────────
@numba.jit(nopython=True, cache=True)
def _maxmaxabs_numba(r: np.ndarray, t: np.ndarray) -> float:
    r = np.abs(r)
    # print(f'{r.shape=}')
    # print(f'{t.shape=}')
    r_max = r[0,0]

    for r0 in r[:,0]:
        if r0 > r_max: r_max = r0

    m = r_max - t[0]

    for i, k in enumerate(t[1:]):
        for r0 in r[:,i]:
            if r0 > r_max: r_max = r0
        m = max(m, r_max - k)
    return m
#───────────────────────────────────────────────────────────────────────
def _maxRMS_python(r: np.ndarray,t: np.ndarray)-> float:
    r *= r
    # print(f'{r.shape=}')
    # print(f'{t.shape=}')
    m = np.sqrt(np.mean(r[:,0])) - t[0]
    for i, k in enumerate(t[1:]):
        m = max(m, np.sqrt(np.mean(r[:,i])) - k)
    return m
#───────────────────────────────────────────────────────────────────────
@numba.jit(nopython=True, cache=True)
def _maxRMS_numba(r: np.ndarray,t: np.ndarray)-> float:
    r *= r
    # print(f'{r.shape=}')
    # print(f'{t.shape=}')
    m = np.sqrt(np.mean(r[:,0])) - t[0]
    for i, k in enumerate(t[1:]):
        m = max(m, np.sqrt(np.mean(r[:,i])) - k)
    return m
#───────────────────────────────────────────────────────────────────────
def _maxsumabs(r: np.ndarray,t: np.ndarray) -> float:
    return np.amax(np.sum(np.abs(r) - t) / t)

errorfunctions = {'maxmaxabs': (_maxmaxabs_python, _maxmaxabs_numba),
                  'maxRMS':(_maxRMS_python, _maxRMS_numba)}
#%%═════════════════════════════════════════════════════════════════════
## ROOT FINDING
def interval(f, x1, y1, x2, y2, fit1):
    '''Returns the last x where f(x)<0'''
    is_debug = _G['debug']
    if is_debug: #─────────────────────────────────────────────────────┐
        print(f'\t{x1=}\t{y1=}')
        print(f'\t{x2=}\t{y2=}')
        _G['mid'], = _G['ax_root'].plot(x1, y1,'.', color = 'blue')
    #──────────────────────────────────────────────────────────────────┘
    # sqrtx1 = x1**0.5
    # sqrtx2 = x2**0.5
    while x2 - x1 > 2:
        if is_debug: #─────────────────────────────────────────────────┐
            wait('\tCalculating new attempt in interval\n')
        #──────────────────────────────────────────────────────────────┘
        # Arithmetic mean between linear estimate and half
        linest = x1 - y1 / (y2 - y1) * (x2 - x1)
        halfest = (x2 + x1) / 2
        # sqrtest1 = sqrtx1 - y1 * (sqrtx2 - sqrtx1) / (y2 - y1)
        # sqrtest1 = sqrtest1*sqrtest1
        # sqrtest2 = int(x1 + (x2 - x1) / (y2 / y1 - 1)**2)
        # x_mid = int((x2 + x1) / 2)
        x_mid = int((linest + halfest) / 2)
        if x_mid == x1:    # To stop repetition in close cases
            x_mid += 1
        elif x_mid == x2:
            x_mid -= 1

        y_mid, fit = f(x_mid)
        if is_debug: #─────────────────────────────────────────────────┐
            print(f'\t{x_mid=}\t{y_mid=}')
            _G['mid'].set_xdata(x_mid)
            _G['mid'].set_ydata(y_mid)
        #──────────────────────────────────────────────────────────────┘
        if y_mid > 0:
            if is_debug: #─────────────────────────────────────────────┐
                wait('\tError over tolerance\n')
                _G['ax_root'].plot(x2, y2,'.', color = 'black')
            #──────────────────────────────────────────────────────────┘
            x2, y2 = x_mid, y_mid
            sqrtx2 = x_mid **0.5
            if is_debug: #─────────────────────────────────────────────┐
                _G['xy2'].set_xdata(x2)
                _G['xy2'].set_ydata(y2)
            #──────────────────────────────────────────────────────────┘
        else:
            if is_debug: #─────────────────────────────────────────────┐
                wait('\tError under tolerance\n')
                _G['ax_root'].plot(x1, y1,'.', color = 'black')
            #──────────────────────────────────────────────────────────┘
            x1, y1, fit1 = x_mid, y_mid, fit
            sqrtx1 = x_mid **0.5
            if is_debug: #─────────────────────────────────────────────┐
                _G['xy1'].set_xdata(x1)
                _G['xy1'].set_ydata(y1)
            #──────────────────────────────────────────────────────────┘

    if x2 - x1 == 2: # Points have only one point in between
        if is_debug: #─────────────────────────────────────────────────┐
            wait('\tPoints have only one point in between\n')
        #──────────────────────────────────────────────────────────────┘
        y_mid, fit = f(x1+1) # Testing that point
        return (x1+1, fit) if (y_mid <0) else (x1, fit1) # If under, give that fit
    else:
        if is_debug: #─────────────────────────────────────────────────┐
            wait('\tPoints have no point in between\n')
        #──────────────────────────────────────────────────────────────┘
        return x1, fit1
#───────────────────────────────────────────────────────────────────────
def droot(f, y1, x2, limit):
    '''Finds the upper limit to interval
    '''
    is_debug = _G['debug']
    x1 = 0
    y2, fit2 = f(x2)
    if is_debug: #─────────────────────────────────────────────────────┐
        _G['xy1'], = _G['ax_root'].plot(x1, y1,'g.')
        _G['xy2'], = _G['ax_root'].plot(x2, y2,'b.')
    #──────────────────────────────────────────────────────────────────┘
    fit1 = None
    while y2 < 0:
        if is_debug: #─────────────────────────────────────────────────┐
            wait('Calculating new attempt in droot\n')
            _G['ax_root'].plot(x1, y1,'.', color = 'black')
        #──────────────────────────────────────────────────────────────┘
        x1, y1, fit1 = x2, y2, fit2
        x2 *= 2
        x2 += 1
        if is_debug: #─────────────────────────────────────────────────┐
            print(f'{limit=}')
            print(f'{x1=}\t{y1=}')
            print(f'{x2=}\t{y2=}')
            _G['xy1'].set_xdata(x1)
            _G['xy1'].set_ydata(y1)
            _G['xy2'].set_xdata(x2)
        #──────────────────────────────────────────────────────────────┘
        if x2 >= limit:
            if is_debug: #─────────────────────────────────────────────┐
                _G['ax_root'].plot([limit, limit], [y1,0],'b.')
            #──────────────────────────────────────────────────────────┘
            y2, fit2 = f(limit)
            if y2<0:
                if is_debug: #─────────────────────────────────────────┐
                    wait('End reached within tolerance\n')
                #──────────────────────────────────────────────────────┘
                return limit, fit2
            else:
                if is_debug: #─────────────────────────────────────────┐
                    wait('End reached outside tolerance\n')
                #──────────────────────────────────────────────────────┘
                x2 = limit
                break
        y2, fit2 = f(x2)
        if is_debug: #─────────────────────────────────────────────────┐
            print(f'{x1=}\t{y1=}')
            print(f'{x2=}\t{y2=}')
            _G['ax_root'].plot(x2, y2,'k.')
            _G['xy2'].set_ydata(y2)
        #──────────────────────────────────────────────────────────────┘
    if is_debug: #─────────────────────────────────────────────────────┐
        _G['xy2'].set_color('red')
        wait('Points for interval found\n')
    #──────────────────────────────────────────────────────────────────┘
    return interval(f, x1, y1, x2, y2, fit1)
#───────────────────────────────────────────────────────────────────────
# @numba.jit(nopython=True,cache=True)
def n_lines(x: np.ndarray, y: np.ndarray, x0: float, y0: np.ndarray, tol: float
            ) -> float:
    '''Estimates number of lines required to fit within error tolerance'''

    if (length := len(x)) > 1:
        inds = sqrtrange_python(0, length - 2) # indices so that x[-1] is not included
        res = (y[-1] - y0) / (x[-1] - x0)*(x[inds] - x0).reshape([-1,1]) - (y[inds] - y0)
        # print(f'sqrtmaxmaxabs {(_maxmaxabs(res, tol)/ tol)** 0.5}')
        # print(f'sqrtmaxsumabs {(_maxsumabs(res, tol)/ tol)** 0.5}')
        # print(f'sqrtmaxRMS {(_maxRMS(res, tol)/ tol)** 0.5}')
        # _maxsumabs(res, tol)
        return 0.5 * (_maxsumabs(res, tol) + 1) ** 0.5 + 1
    else:
        return 1.
#───────────────────────────────────────────────────────────────────────

def _get_f2zero(x, y, x0, y0, sqrtrange, f_fit, errorfunction, tol):
    def f2zero(i: int) -> tuple:
        '''Function such that i is optimal when f2zero(i) = 0'''
        inds = sqrtrange(0, i)
        residuals, fit = f_fit(x[inds], y[inds], x0, y0)
        return errorfunction(residuals, tol), fit
    return f2zero
#───────────────────────────────────────────────────────────────────────
def _get_f2zero_debug(x, y, x0, y0, sqrtrange, f_fit, errorfunction, tol):
    def f2zero_debug(i: int) -> tuple:
        '''Function such that i is optimal when f2zero(i) = 0'''
        inds = sqrtrange(0, i)
        residuals, fit = f_fit(x[inds], y[inds], x0, y0)

        print(f'\t\t{i=}\t{residuals.shape=}')
        print(f'\t\tx {x[inds][0]} - {x[inds][-1]}')
        indices_all = np.arange(-1, i + 1) + _G['start']
        _G['x_plot'] = _G['x'][indices_all]
        _G['y_plot'] = _G['fyc'](fit, _G['x_plot'])
        _G['line_fit'].set_xdata(_G['x_plot'])
        _G['line_fit'].set_ydata(_G['y_plot'])
        # print(f'{_G["y_plot"].shape=}')
        # print(f'{_G["y"][indices_all].shape=}')
        res_all = _G['y_plot'] - _G['y'][indices_all].flatten()
        print(f'\t\t{res_all.shape=}')
        _G['ax_res'].clear()
        _G['ax_res'].grid()
        _G['ax_res'].axhline(color = 'red', linestyle = '--')
        _G['ax_res'].set_ylabel('Residual relative to tolerance')
        _G['ax_res'].plot(indices_all - _G['start'], np.abs(res_all) / tol -1,
                            '.', color = 'blue', label = 'ignored')
        _G['ax_res'].plot(inds, np.abs(residuals) / tol-1,
                            'o', color = 'red', label = 'sampled')
        _G['ax_res'].legend()
        wait('\t\tFitting\n')
        return errorfunction(residuals, tol), fit
    return f2zero_debug
#───────────────────────────────────────────────────────────────────────
def gen_f2zero(*args):
    '''Generates function for the root finder'''
    return _get_f2zero_debug(*args) if _G['debug'] else _get_f2zero(*args)
###═════════════════════════════════════════════════════════════════════
### BLOCK COMPRESSION
def LSQ10(x: np.ndarray, y: np.ndarray, tol = 1e-2, initial_step = None,
          errorfunction = 'maxmaxabs', use_numba = 0, fitset = 'Poly10') -> tuple:
    '''Compresses the data of 1-dimensional system of equations
    i.e. single wait variable and one or more output variable
    '''
    is_debug = _G['debug']
    if _G['timed']:
        _G['t_start'] = time.perf_counter()
    start = 1 # Index of starting point for looking for optimum
    end = len(x) - 1 # Number of unRecord datapoints -1, i.e. the last index
    limit = end - start
    fit = None
    
    x = to_ndarray(x)
    y = to_ndarray(y, (len(x), -1))

    tol = to_ndarray(tol, y[0].shape)
    start_y1 = - np.amax(tol) # Starting value for discrete root calculation
    sqrtrange = _sqrtrange[use_numba]
    if isinstance(errorfunction, str):
        errorfunction = errorfunctions[errorfunction][use_numba]
    if isinstance(fitset, str):
        fitset = fitsets[fitset]
    f_fit = fitset.fit[use_numba]
    fyc = fitset.y_from_fit

    x_c, y_c = [x[0]], [np.array(y[0])]
    # Estimation for the first offset
    if initial_step is None:
        mid = end // 2
        offset = round(limit / n_lines(x[1:mid], y[1:mid], x[0], y[0], tol))
    else:
        offset = initial_step
            
    if is_debug: #─────────────────────────────────────────────────────┐
        _G.update({'x': x,
                   'y': y,
                   'tol': tol,
                   'fyc': fyc,
                   'start': start})

        _G['fig'], axs = plt.subplots(3,1)
        for ax in axs:
            ax.grid()
        _G['ax_data'], _G['ax_res'], _G['ax_root'] = axs

        _G['ax_data'].fill_between(_G['x'].flatten(), (_G['y'] - tol).flatten(), (_G['y'] + _G['tol']).flatten(), alpha=.3, color = 'blue')

        _G['line_fit'], = _G['ax_data'].plot(0,0,'-',color = 'orange')
        _G['ax_res'].axhline(color = 'red', linestyle = '--')
        _G['ax_root'].set_ylabel('Tolerance left')

        plt.ion()
        plt.show()
        wait('Starting\n')
        print(f'{offset=}')
    #──────────────────────────────────────────────────────────────────┘
    for _ in range(end): # Prevents infinite loop in case error
        offset, fit = droot(gen_f2zero(x[start:], y[start:], x_c[-1], y_c[-1],
                            sqrtrange, f_fit, errorfunction, tol),
                            start_y1, offset, limit)
        step = offset + 1
        if fit is None: raise RuntimeError('Fit not found')
        if is_debug:
            print(f'{start=}\t{offset=}\t{end=}\t')
            print(f'{fit=}')
            _G['ax_root'].clear()
            _G['ax_root'].grid()
            _G['ax_root'].set_ylabel('Maximum residual')
            _G['ax_data'].plot(_G['x_plot'], _G['y_plot'], color = 'red')

        start += step # Start shifted by the number Record and the
        if start > end:
            break
        x_c.append(x[start - 1])
        y_c.append(fyc(fit, x_c[-1]))
        limit -= step
        offset = min(limit, offset) # Setting up to be next estimation

        if is_debug: #─────────────────────────────────────────────────┐
            _G['start'] = start
            if len(x_c) == 40:
                print(f'\nTrigger{40 * "═"}\n')
                time.sleep(1)
                wait()
            _G['ax_data'].plot(x_c[-1], y_c[-1],'go')
            wait('Next iteration\n')
        #──────────────────────────────────────────────────────────────┘
    else:
        raise StopIteration('Maximum number of iterations reached')
    # Last data point is same as in the unRecord data
    x_c.append(x[-1])
    y_c.append(y[-1])

    if _G['timed']:
        _G['runtime'] = time.perf_counter() - _G['t_start']
    
    if is_debug:
        plt.ioff()
    return to_ndarray(x_c), to_ndarray(y_c)
###═════════════════════════════════════════════════════════════════════
def pick(x, y, tol=1e-2, mins=30, verbosity=0, is_timed=False, use_numba = 0):
    '''Returns inds of data points to select.
    Should be faster than LSQ based'''

    if is_timed: t_start = time.perf_counter()

    zero = 1
    end = len(x)- 1 - zero
    estimate = int(end/n_lines(x,y,x[0],y[0],tol) )+1
    inds = [0]
    errorfunction = errorfunctions['maxmaxabs'][use_numba]
    sqrtrange = _sqrtrange[use_numba]

    x = x.reshape([-1, 1])
    y = y.reshape([len(x), -1])

    if not isinstance(tol, (list, np.ndarray)):
        tol = [tol] * y.shape[1]
    tol = np.array(tol)
    #───────────────────────────────────────────────────────────────────
    def f2zero(n, xs, ys, tol):

        inds = sqrtrange(zero, n)

        a = (y[n+zero] - y[zero])/(x[n+zero] - x[zero])
        b = y[zero] - a * xs[zero]
        residuals = a * x[inds] + b - y[inds]

        return errorfunction(residuals, tol), None
    #───────────────────────────────────────────────────────────────────
    while end > 0:
        estimate = int((end + end/(n_lines(x[zero:], y[zero:], 
                                           x[inds[-1]], y[inds[-1]], tol)))/2)
        estimate = min(end, estimate)
        end, _ = droot(f2zero,-tol, estimate, end)
        end += 1
        zero += end
        end -= end # WAT.
        
        inds.append(zero-1)

    if is_timed: t = time.perf_counter()-t_start
    if verbosity>0: 
        text = 'Length of Record array\t%i'%len(inds)
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
        text = 'Length of Record array\t%i'%len(inds)
        text += '\nCompression factor\t%.3f %%' % 100*len(inds)/len(x)
        if is_timed: text += '\nCompression time\t%.1f ms' % (t*1e3)
        print(text)
    return inds
###═════════════════════════════════════════════════════════════════════
### STREAM COMPRESSION
class _StreamRecord(collections.abc.Sized):
    """Class for doing stream compression for data of 1-dimensional
    system of equations 
    i.e. single wait variable and one or more output variable
    """
    def __init__(self, x0: float, y0: np.ndarray, tol: np.ndarray, errorfunction: str, use_numba: int, fitset, x2):
        self.is_debug = _G['debug']
        if _G['timed']: _G['t_start'] = time.perf_counter()
        self.xb, self.yb = [], [] # Buffers for yet-to-be-recorded data
        self.xc, self.yc = [x0], [y0]
        self.x1 = 0 # Index of starting point for looking for optimum
        self.x2 = x2
        self.tol = tol
        self.start_y1 = -np.amax(tol) # Default starting value
        self.state = 'open' # The object is ready to accept more values
        self.errorfunction = errorfunction
        self.fitset = fitset
        self.f_fit = self.fitset.fit[use_numba]
        self.sqrtrange = _sqrtrange[use_numba]
        self.fyc = self.fitset.y_from_fit
        self.limit = -1 # Last index of the buffer

        self._lenb = 0 # length of the buffer
        self._lenc = 1 # length of the Record points

        self.y1 = -self.tol # Initialising
        if self.is_debug: #────────────────────────────────────────────┐
            _G['fig'], axs = plt.subplots(3,1)
            for ax in axs:
                ax.grid()
            _G['ax_data'], _G['ax_res'], _G['ax_root'] = axs
            _G['line_buffer'], = _G['ax_data'].plot(0, 0, 'b-',
                                                    label = 'buffer')
            _G['line_fit'], = _G['ax_data'].plot(0, 0, '-', color = 'orange',
                                                 label = 'fit')

            _G['ax_root'].set_ylabel('Tolerance left')

            plt.ion()
            plt.show()
        #──────────────────────────────────────────────────────────────┘
    #───────────────────────────────────────────────────────────────────
    def _f2zero(self, i: int) -> tuple:
        '''Function such that i is optimal when f2zero(i) = 0'''

        inds = self.sqrtrange(0, i)
        residuals, fit = self.f_fit(self.xb[inds], self.yb[inds],
                                    self.xc[-1], self.yc[-1])
        if self.is_debug: #────────────────────────────────────────────┐
            if residuals.shape != (len(inds), len(self.yb[-1])):
                raise ValueError(f'{residuals.shape=}')
            print(f'\t\t{i=}\t{residuals.shape=}')
            print(f'\t\tx {self.xb[inds][0]} - {self.xb[inds][-1]}')
            indices_all = np.arange(0, i + 1)
            _G['ax_data'].plot(self.xb[i], self.yb[i], 'k.')
            _G['x_plot'] = self.xb[indices_all]
            _G['y_plot'] = self.fyc(fit, _G['x_plot'])
            _G['line_fit'].set_xdata(_G['x_plot'])
            _G['line_fit'].set_ydata(_G['y_plot'])
            res_all = _G['y_plot'] - self.yb[indices_all].flatten()
            _G['ax_res'].clear()
            _G['ax_res'].grid()
            _G['ax_res'].set_ylabel('Residual relative to tolerance')
            _G['ax_res'].plot(indices_all, np.abs(res_all) / self.tol -1,
                             'b.', label = 'ignored')
            _G['ax_res'].plot(inds, np.abs(residuals) / self.tol - 1,
                             'ro', label = 'sampled')
            _G['ax_res'].legend()
            wait('\t\tFitting\n')
        #──────────────────────────────────────────────────────────────┘
        return self.errorfunction(residuals, self.tol), fit
    #───────────────────────────────────────────────────────────────────
    def squeeze_buffer(self):
        '''Compresses the buffer by one step'''
        #──────────────────────────────────────────────────────────────┘
        offset, fit = interval(self._f2zero, self.x1, self.y1,
                               self.x2, self.y2, self.fit1)
        if self.is_debug: #────────────────────────────────────────────┐
            if fit is None: raise RuntimeError('Fit not found')
            print(f'{offset=}')
            print(f'{fit=}')
            _G['ax_root'].clear()
            _G['ax_root'].grid()
            _G['ax_root'].set_ylabel('Maximum residual')
            _G['ax_data'].plot(_G['x_plot'], _G['y_plot'], color = 'red')
        #──────────────────────────────────────────────────────────────┘
        self.x1, self.y1, step = 0, self.start_y1, offset + 1
        
        self.xc.append(self.xb[offset])
        self.yc.append(self.fyc(fit, self.xc[-1]))

        self.limit -= step
        self._lenb -= step
        self._lenc += 1
        if self.is_debug: #────────────────────────────────────────────┐
            if self._lenc == 40:
                print(f'\nTrigger{40 * "═"}\n')
                time.sleep(1)
                wait()
        #──────────────────────────────────────────────────────────────┘
        self.x2 = offset # Approximation

        self.xb, self.yb = self.xb[step:], self.yb[step:]
        if self.xc[-1] == self.xb[0]: raise IndexError('derp')
        
    #───────────────────────────────────────────────────────────────────
    def __call__(self, x_raw, y_raw):
        self.xb.append(x_raw)
        self.yb.append(to_ndarray(y_raw, (-1,)))
        self.limit += 1
        self._lenb += 1

        if self.is_debug: #────────────────────────────────────────────┐
            _G['line_buffer'].set_xdata(self.xb)
            _G['line_buffer'].set_ydata(self.yb)
        #──────────────────────────────────────────────────────────────┘
        if  self.limit >= self.x2: #───────────────────────────────────┐
            # Converting to numpy arrays for computations
            self.xb = to_ndarray(self.xb)
            self.yb = to_ndarray(self.yb, (self._lenb, -1))
            
            if self.is_debug: #────────────────────────────────────────┐
                if self.xb.shape != (self._lenb,):
                    raise ValueError(f'{self.xb.shape=}')

                if self.yb.shape != (self._lenb, len(self.yc[0])):
                    raise ValueError(f'{self.yb.shape=}')
            #──────────────────────────────────────────────────────────┘
            self.y2, self.fit2 = self._f2zero(self.x2)

            if self.is_debug: #────────────────────────────────────────┐
                _G['xy1'], = _G['ax_root'].plot(self.x1, self.y1,'g.')
                _G['xy2'], = _G['ax_root'].plot(self.x2, self.y2,'b.')
            #──────────────────────────────────────────────────────────┘
            if self.y2 < 0: #──────────────────────────────────────────┐
                if self.is_debug: #────────────────────────────────────┐
                    wait('Calculating new attempt in end\n')
                    _G['ax_root'].plot(self.x1, self.y1,'.', color = 'black')
                #──────────────────────────────────────────────────────┘
                self.x1, self.y1, self.fit1 = self.x2, self.y2, self.fit2
                self.x2 *= 2
                self.x2 += 1

                if self.is_debug: #────────────────────────────────────┐
                    print(f'{self.limit=}')
                    print(f'{self.x1=}\t{self.y1=}')
                    print(f'{self.x2=}\t{self.y2=}')
                    _G['xy1'].set_xdata(self.x1)
                    _G['xy1'].set_ydata(self.y1)
                    _G['xy2'].set_xdata(self.x2)
                #──────────────────────────────────────────────────────┘
            else: # Squeezing the buffer
                if self.is_debug: #────────────────────────────────────┐
                    _G['xy2'].set_color('red')
                    wait('Points for interval found\n')
                    print(f'{self._lenc=}')
                #──────────────────────────────────────────────────────┘
                self.squeeze_buffer()
            #──────────────────────────────────────────────────────────┘
            
            # Converting back to lists
            self.xb, self.yb = list(self.xb.flatten()), list(self.yb)
            if self.is_debug: #────────────────────────────────────────┐
                _G['ax_data'].plot(self.xc[-1], self.yc[-1], 'go')
                wait('Next iteration\n')
                if self.yb[-1].shape != (1,):
                    raise ValueError(f'{self.yb[-1].shape=}')
                if self.yc[-1].shape != (1,):
                    raise ValueError(f'{self.yc[-1].shape=}')
            #──────────────────────────────────────────────────────────┘
        #──────────────────────────────────────────────────────────────┘
        return self._lenc, self._lenb
    #───────────────────────────────────────────────────────────────────
    def __len__(self):
        return self._lenc + self._lenb
    #───────────────────────────────────────────────────────────────────
    def __str__(self):
        return f'{self.x=} {self.y=} {self.tol=}'
    #───────────────────────────────────────────────────────────────────
    def close(self):
        self.state = 'closing'

        # Converting to numpy arrays for computations
        self.xb = to_ndarray(self.xb)
        self.yb = to_ndarray(self.yb, (self._lenb, -1))
        # print(f'Calling f2zero with {self.limit=}')
        self.x2 = min(self.x2, self.limit)
        self.y2, self.fit2 = self._f2zero(self.x2)
        

        while self.y2 > 0: #───────────────────────────────────────────┐
            self.squeeze_buffer()
            self.x2 = min(self.x2, self.limit)
            self.y2, self.fit2 = self._f2zero(self.x2)
        #──────────────────────────────────────────────────────────────┘
        self.xc.append(self.xb[-1])
        self.yc.append(to_ndarray(self.yb[-1], (1,)))
        
        if self.is_debug: plt.ioff()
        # Final packing and cleaning
        self.x = to_ndarray(self.xc, (self._lenc+1,))
        self.y = to_ndarray(self.yc, (self._lenc+1, -1))
        for key in tuple(self.__dict__):
            if key not in {'x', 'y', 'state', 'tol'}:
                del self.__dict__[key]

        self.state = 'closed'
        if _G['timed']: _G['runtime'] = time.perf_counter() - _G['t_start']
    #───────────────────────────────────────────────────────────────────
###═════════════════════════════════════════════════════════════════════
class Stream():
    '''Context manager for stream compression of data of 
    1 dimensional system of equations'''
    def __init__(self, x0, y0, tol = 1e-2, initial_step = None,
                 errorfunction = 'maxmaxabs', use_numba = 0, fitset = 'Poly10'):
        self.x0            = x0
        # Variables are columns, e._G. 3xn
        self.y0            = to_ndarray(y0, (-1,))
        self.tol           = to_ndarray(tol, self.y0.shape)
        
        if isinstance(errorfunction, str): #───────────────────────────┐
            self.errorfunction = errorfunctions[errorfunction][use_numba]
        else:
            self.errorfunction = errorfunction
        #──────────────────────────────────────────────────────────────┘
        if isinstance(fitset, str): #──────────────────────────────────┐
            self.fitset = fitsets[fitset]
        else:
            self.fitset = fitset
        #──────────────────────────────────────────────────────────────┘
        self.use_numba     = use_numba
        self.x2            = 100 if initial_step is None else initial_step
    #───────────────────────────────────────────────────────────────────
    def __enter__(self):
        self.record = _StreamRecord(self.x0, self.y0, self.tol,
                                          self.errorfunction, self.use_numba, self.fitset, self.x2)
        return self.record
    #───────────────────────────────────────────────────────────────────
    def __exit__(self, exc_type, exc_value, traceback):
        self.record.close()
#%%═════════════════════════════════════════════════════════════════════
# CUSTOM FUNCTIONS

class Poly10:
    """Builtin group of functions for doing the compression"""
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    def fit_python(x: np.ndarray, y: np.ndarray, x0, y0: np.ndarray) -> tuple:
        '''Takes block of data, previous fitting parameters and calculates next fitting parameters'''

        Dx = x - x0
        Dy = y - y0
        a = Dx @ Dy / Dx.dot(Dx)
        b = y0 - a * x0
        # print(f'{x.shape=}')
        # print(f'{y.shape=}')
        # print(f'{y0.shape=}')
        # print(f'{x0=}')
        # print(f'{Dx.shape=}')
        # print(f'{Dy.shape=}')
        # print(f'{a.shape=}')
        # print(f'{b.shape=}')
        return (np.outer(Dx, a) - Dy, (a,  b))
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def fit_numba(x: np.ndarray, y: np.ndarray, x0, y0: np.ndarray) -> tuple:
        '''Takes block of data, previous fitting parameters and calculates next fitting parameters'''

        Dx = x - x0
        Dy = y - y0
        a = Dx.T @ Dy / Dx.T.dot(Dx)
        b = y0 - a * x0
        # print(f'{x.shape=}')
        # print(f'{y.shape=}')
        # print(f'{y0.shape=}')
        # print(f'{x0=}')
        # print(f'{Dx.shape=}')
        # print(f'{Dy.shape=}')
        # print(f'{a.shape=}')
        # print(f'{b.shape=}')
        return (a * Dx - Dy, (a,  b))
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    def y_from_fit(fit: tuple, x: np.ndarray) -> np.ndarray:
        '''Converts the fitting parameters and x to storable y values'''
        # print(f'{fit[0].shape=}')
        # print(f'{fit[1].shape=}')
        # print(f'{x.shape=}')
        return (fit[0] * x + fit[1]).flatten()
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    def fit_from_end(ends: np.ndarray) -> tuple:
        '''Takes storable y values and return fitting parameters'''
        a = (ends[1,1:] - ends[0,1:]) / (ends[1,0] - ends[0,0])
        b = ends[0,1:] - a * ends[0,0]
        return a, b
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    def decompress(x_compressed: np.ndarray, fit_array: np.ndarray):
        '''Takes array of fitting parameters and constructs whole function'''
        rows, columns = fit_array.shape
        def _iteration(x, low = 0):
            index = bisect_left(x_compressed, x, lo = low, hi = rows)
            if index == 0:
                return fit_array[0], index
            try:
                x1, x2 = x_compressed[index-1:(index + 1)]
                y1, y2 = fit_array[index-1:(index + 1)]
                return ((y2 - y1) / (x2 - x1) * (x - x1) + y1), index
            except ValueError:
                return fit_array[index], index
        #───────────────────────────────────────────────────────────────
        def interpolator(x_input): #────────────────────────────────────
            if isinstance(x_input, np.ndarray):
                out = np.full((len(x_input), columns), np.nan)
                i_c = 0
                for i_out, x in enumerate(x_input): 
                    out[i_out], i_c = _iteration(x, i_c)
                return out
            else:
                return _iteration(x_input)[0]
        #───────────────────────────────────────────────────────────────
        return interpolator
    #───────────────────────────────────────────────────────────────────
    fit = (fit_python, fit_numba)
#%%═════════════════════════════════════════════════════════════════════
# WRAPPING

compressors = {'LSQ10': LSQ10,
               'pick': pick,
               'split': split}
decompressors = {'Poly10': Poly10.decompress}
#───────────────────────────────────────────────────────────────────────
def compress(*args, compressor = 'LSQ10', **kwargs):
    '''Wrapper for easier selection of compression method'''
    if isinstance(compressor, str): 
        try:
            compressor = compressors[compressor]
        except KeyError:
            raise NotImplementedError(f'{compressor} not in the dictionary of builtin compressors')
    return compressor(*args, **kwargs)
#───────────────────────────────────────────────────────────────────────
def decompress(x, y, decompressor = 'Poly10', **kwargs):
    '''Wrapper for easier selection of compression method'''
    if isinstance(decompressor, str):
        try:
            decompressor = decompressors[decompressor]
        except KeyError:
            raise NotImplementedError("Method not in the dictionary of methods")
    return decompressor(x, y,**kwargs)
#%%═════════════════════════════════════════════════════════════════════
# HACKS
# A hack to make the package callable
class Pseudomodule(types.ModuleType):
    """Class that wraps the individual plotting functions
    an allows making the module callable"""
    @staticmethod
    def __call__(*args, compressor = 'LSQ10', decompressor = 'Poly10', **kwargs):
        '''Wrapper for easier for combined compression and decompression'''

        return decompress(*compress(*args, compressor = compressor, **kwargs),
                          decompressor = decompressor)

#───────────────────────────────────────────────────────────────────────
fitsets = {'Poly10': Poly10}
#%%═════════════════════════════════════════════════════════════════════
# Here the magic happens for making the API module itself also callable
sys.modules[__name__].__class__ = Pseudomodule