#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
API
========================================================================

Connection point for all package utilities to be provided
'''

import collections
import numba
import numpy as np
import sys
import time
import types
from bisect import bisect_left
import matplotlib.pyplot as plt

from . import GLOBALS
from . import reference as ref # Careful with this circular import
from . import auxiliaries as aux
from .auxiliaries import to_ndarray, wait
from . import models
from . import GLOBALS 
# This global dictionary G is for passing some telemtery and debug arguments
global G
G = GLOBALS.dictionary

_sqrtrange = (aux.sqrtrange_python, aux.sqrtrange_numba)
#%%═════════════════════════════════════════════════════════════════════
## ERROR TERM
def _maxmaxabs_python(residuals: np.ndarray, tolerance: np.ndarray) -> float:
    '''Python version'''
    residuals = np.abs(residuals)
    # print(f'{residuals.shape=}')
    # print(f'{tolerance.shape=}')
    r_max = residuals[0,0] # Initialising

    # Going through first column to initialise maximum value
    for r0 in residuals[:,0]:
        if r0 > r_max: r_max = r0
    dev_max = r_max - tolerance[0]

    for i, k in enumerate(tolerance[1:]):
        for r0 in residuals[:,i]:
            if r0 > r_max: r_max = r0
        deviation = r_max - k
        if deviation > dev_max: dev_max = deviation
    return dev_max
#───────────────────────────────────────────────────────────────────────
@numba.jit(nopython=True, cache=True, fastmath = True)
def _maxmaxabs_numba(residuals: np.ndarray, tolerance: np.ndarray) -> float:
    residuals = np.abs(residuals)
    # print(f'{residuals.shape=}')
    # print(f'{tolerance.shape=}')
    r_max = residuals[0,0] # Initialising

    # Going through first column to initialise maximum value
    for r0 in residuals[:,0]:
        if r0 > r_max: r_max = r0
    dev_max = r_max - tolerance[0]

    for i, k in enumerate(tolerance[1:]):
        for r0 in residuals[:,i]:
            if r0 > r_max: r_max = r0
        deviation = r_max - k
        if deviation > dev_max: dev_max = deviation
    return dev_max
#───────────────────────────────────────────────────────────────────────
def _maxRMS_python(residuals: np.ndarray, tolerance: np.ndarray)-> float:
    residuals *= residuals
    # print(f'{residuals.shape=}')
    # print(f'{tolerance.shape=}')
    m = np.sqrt(np.mean(residuals[:,0])) - tolerance[0]
    for i, k in enumerate(tolerance[1:]):
        m = max(m, np.sqrt(np.mean(residuals[:,i])) - k)
    return m
#───────────────────────────────────────────────────────────────────────
@numba.jit(nopython=True, cache=True)
def _maxRMS_numba(residuals: np.ndarray, tolerance: np.ndarray)-> float:
    residuals *= residuals
    # print(f'{residuals.shape=}')
    # print(f'{tolerance.shape=}')
    m = np.sqrt(np.mean(residuals[:,0])) - tolerance[0]
    for i, k in enumerate(tolerance[1:]):
        m = max(m, np.sqrt(np.mean(residuals[:,i])) - k)
    return m
#───────────────────────────────────────────────────────────────────────
def _maxsumabs(residuals: np.ndarray,tolerance: np.ndarray) -> float:
    return np.amax(np.sum(np.abs(residuals) - tolerance) / tolerance)
#───────────────────────────────────────────────────────────────────────
_errorfunctions = {'maxmaxabs': (_maxmaxabs_python, _maxmaxabs_numba),
                  'maxRMS':(_maxRMS_python, _maxRMS_numba)}
#───────────────────────────────────────────────────────────────────────
def get_errorfunction(name, use_numba, tol):
    _errorfunction = _errorfunctions[name][use_numba]
    return lambda residuals: _errorfunction(residuals, tol)
#%%═════════════════════════════════════════════════════════════════════
## ROOT FINDING
def interval(f, x1, y1, x2, y2, fit1):
    '''Returns the last x where f(x)<0'''
    is_debug = G['debug']
    if is_debug: #─────────────────────────────────────────────────────┐
        print(f'\t{x1=}\t{y1=}')
        print(f'\t{x2=}\t{y2=}')
        G['mid'], = G['ax_root'].plot(x1, y1,'.', color = 'blue')
    #──────────────────────────────────────────────────────────────────┘
    # sqrtx1 = x1**0.5
    # sqrtx2 = x2**0.5
    while x2 - x1 > 2:
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

        y_mid, fit2 = f(x_mid)
        if is_debug: #─────────────────────────────────────────────────┐
            print(f'\t{x_mid=}\t{y_mid=}')
            G['mid'].set_xdata(x_mid)
            G['mid'].set_ydata(y_mid)
        #──────────────────────────────────────────────────────────────┘
        if y_mid > 0:
            if is_debug: #─────────────────────────────────────────────┐
                wait('\tError over tolerance\n')
                G['ax_root'].plot(x2, y2,'.', color = 'black')
            #──────────────────────────────────────────────────────────┘
            x2, y2 = x_mid, y_mid
            sqrtx2 = x_mid **0.5
            if is_debug: #─────────────────────────────────────────────┐
                G['xy2'].set_xdata(x2)
                G['xy2'].set_ydata(y2)
            #──────────────────────────────────────────────────────────┘
        else:
            if is_debug: #─────────────────────────────────────────────┐
                wait('\tError under tolerance\n')
                G['ax_root'].plot(x1, y1,'.', color = 'black')
            #──────────────────────────────────────────────────────────┘
            x1, y1, fit1 = x_mid, y_mid, fit2
            sqrtx1 = x_mid ** 0.5
            if is_debug: #─────────────────────────────────────────────┐
                G['xy1'].set_xdata(x1)
                G['xy1'].set_ydata(y1)
            #──────────────────────────────────────────────────────────┘
    if x2 - x1 == 2: # Points have only one point in between
        y_mid, fit2 = f(x1+1) # Testing that point
        return (x1+1, fit2) if (y_mid <0) else (x1, fit1) # If under, give that fit
    else:
        return x1, fit1
#───────────────────────────────────────────────────────────────────────
def droot(f, y1, x2, limit):
    '''Finds the upper limit to interval
    '''
    is_debug = G['debug']
    x1 = 0
    y2, fit2 = f(x2)
    fit1 = None
    if is_debug: #─────────────────────────────────────────────────────┐
        G['xy1'], = G['ax_root'].plot(x1, y1,'g.')
        G['xy2'], = G['ax_root'].plot(x2, y2,'b.')
    #──────────────────────────────────────────────────────────────────┘
    while y2 < 0:
        if is_debug: #─────────────────────────────────────────────────┐
            wait('Calculating new attempt in droot\n')
            G['ax_root'].plot(x1, y1,'k.')
        #──────────────────────────────────────────────────────────────┘
        x1, y1, fit1 = x2, y2, fit2
        x2 *= 2
        x2 += 1
        if is_debug: #─────────────────────────────────────────────────┐
            print(f'{limit=}')
            print(f'{x1=}\t{y1=}')
            print(f'{x2=}\t{y2=}')
            G['xy1'].set_xdata(x1)
            G['xy1'].set_ydata(y1)
            G['xy2'].set_xdata(x2)
        #──────────────────────────────────────────────────────────────┘
        if x2 >= limit:
            if is_debug: #─────────────────────────────────────────────┐
                G['ax_root'].plot([limit, limit], [y1,0],'b.')
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
            G['ax_root'].plot(x1, y1,'k.')
            print(f'{x1=}\t{y1=}')
            print(f'{x2=}\t{y2=}')
            G['ax_root'].plot(x2, y2,'k.')
            G['xy2'].set_ydata(y2)
        #──────────────────────────────────────────────────────────────┘
    if is_debug: #─────────────────────────────────────────────────────┐
        G['xy2'].set_color('red')
        wait('Points for interval found\n')
    #──────────────────────────────────────────────────────────────────┘
    return interval(f, x1, y1, x2, y2, fit1)
#───────────────────────────────────────────────────────────────────────
# @numba.jit(nopython=True,cache=True)
def n_lines(x: np.ndarray, y: np.ndarray, x0: float, y0: np.ndarray, tol: float
            ) -> float:
    '''Estimates number of lines required to fit within error tolerance'''

    if (length := len(x)) > 1:
        inds = _sqrtrange[0](length - 2) # indices so that x[-1] is not included
        res = (y[-1] - y0) / (x[-1] - x0)*(x[inds] - x0).reshape([-1,1]) - (y[inds] - y0)

        reference = _maxsumabs(res, tol)
        if reference < 0: reference = 0
        return 0.5 * reference ** 0.5 + 1
    else:
        return 1.
#───────────────────────────────────────────────────────────────────────
def _get_f2zero(x, y, x0, y0, sqrtrange, f_fit, errorfunction):
    def f2zero(i: int) -> tuple:
        '''Function such that i is optimal when f2zero(i) = 0'''
        inds = sqrtrange(i)
        residuals, fit = f_fit(x[inds], y[inds], x0, y0)
        return errorfunction(residuals), fit
    return f2zero
#───────────────────────────────────────────────────────────────────────
def _get_f2zero_debug(x, y, x0, y0, sqrtrange, f_fit, errorfunction):
    def f2zero_debug(i: int) -> tuple:
        '''Function such that i is optimal when f2zero(i) = 0'''
        inds = sqrtrange(i)
        residuals, fit = f_fit(x[inds], y[inds], x0, y0)
        if len(residuals) == 1:
            print(f'\t\t{residuals=}')
        print(f'\t\tstart = {G["start"]} end = {i + G["start"]} points = {i + 1}')
        print(f'\t\tx0\t{x0}\n\t\tx[0]\t{x[inds][0]}\n\t\tx[-1]\t{x[inds][-1]}\n\t\txstart = {G["x"][G["start"]]}')
        indices_all = np.arange(-1, i + 1) + G['start']
        G['x_plot'] = G['x'][indices_all]
        G['y_plot'] = G['fyc'](fit, G['x_plot'])
        G['line_fit'].set_xdata(G['x_plot'])
        G['line_fit'].set_ydata(G['y_plot'])
        # print(f'{G["y_plot"].shape=}')
        # print(f'{G["y"][indices_all].shape=}')
        res_all = G['y_plot'] - G['y'][indices_all].flatten()
        print(f'\t\t{residuals.shape=}\n\t\t{res_all.shape=}')
        G['ax_res'].clear()
        G['ax_res'].grid()
        G['ax_res'].axhline(color = 'red', linestyle = '--')
        G['ax_res'].set_ylabel('Residual relative to tolerance')
        G['ax_res'].plot(indices_all - G['start'], np.abs(res_all) / G['tol'] -1,
                            '.', color = 'blue', label = 'ignored')
        G['ax_res'].plot(inds, np.abs(residuals) / G['tol']-1,
                            'o', color = 'red', label = 'sampled')
        G['ax_res'].legend(loc = 'lower right')
        wait('\t\tFitting\n')
        return errorfunction(residuals), fit
    return f2zero_debug
#───────────────────────────────────────────────────────────────────────
def gen_f2zero(*args):
    '''Generates function for the root finder'''
    return _get_f2zero_debug(*args) if G['debug'] else _get_f2zero(*args)
###═════════════════════════════════════════════════════════════════════
### BLOCK COMPRESSION
def LSQ10(x_in: np.ndarray, y_in: np.ndarray, tol = 1e-2, initial_step = None,
          errorfunction = 'maxmaxabs', use_numba = 0, fitset = 'Poly10') -> tuple:
    '''Compresses the data of 1-dimensional system of equations
    i.e. single wait variable and one or more output variable
    '''
    is_debug = G['debug']
    if G['timed']:
        G['t_start'] = time.perf_counter()
    start = 1 # Index of starting point for looking for optimum
    end = len(x_in) - 1 # Number of unRecord datapoints -1, i.e. the last index
    limit = end - start

    x = to_ndarray(x_in)
    y = to_ndarray(y_in, (len(x), -1))

    tol = to_ndarray(tol, y[0].shape)
    start_y1 = - np.amax(tol) # Starting value for discrete root calculation
    sqrtrange = _sqrtrange[use_numba]
    if isinstance(errorfunction, str):
        errorfunction = get_errorfunction(errorfunction, use_numba, tol)
    if isinstance(fitset, str):
        fitset = fitsets[fitset]
    f_fit = fitset.fit[use_numba]
    fyc = fitset.y_from_fit

    xc, yc = [x[0]], [y[0]]

    # Estimation for the first offset
    if initial_step is None:
        mid = end // 2
        offset = round(limit / n_lines(x[1:mid], y[1:mid], x[0], y[0], tol))
    else:
        offset = initial_step

    if is_debug: #─────────────────────────────────────────────────────┐
        G.update({'x': x,
                   'y': y,
                   'tol': tol,
                   'fyc': fyc,
                   'start': start})

        G['fig'], axs = plt.subplots(3,1)
        for ax in axs:
            ax.grid()
        G['ax_data'], G['ax_res'], G['ax_root'] = axs

        G['ax_data'].fill_between(G['x'].flatten(), (G['y'] - tol).flatten(), (G['y'] + G['tol']).flatten(), alpha=.3, color = 'blue')

        G['line_fit'], = G['ax_data'].plot(0, 0, '-', color = 'orange')
        G['ax_res'].axhline(color = 'red', linestyle = '--')
        G['ax_root'].set_ylabel('Tolerance left')
        G['ax_root'].axhline(color = 'red', linestyle = '--')

        plt.ion()
        plt.show()
        wait('Starting\n')
        print(f'{offset=}')
    #──────────────────────────────────────────────────────────────────┘
    for _ in range(end): # Prevents infinite loop in case error
        if x[start-1] != xc[-1]:
            raise IndexError(f'Indices out of sync {start}')
        offset, fit = droot(gen_f2zero(x[start:], y[start:], xc[-1], yc[-1],
                                       sqrtrange, f_fit, errorfunction),
                            start_y1, offset, limit)
        step = offset + 1
        start += step # Start shifted by the number Record and the
        if start > end:
            break
        xc.append(x[start - 1])
        if is_debug: #─────────────────────────────────────────────────┐
            print(f'{start=}\t{offset=}\t{end=}\t')
            print(f'{fit=}')
            G['ax_root'].clear()
            G['ax_root'].grid()
            G['ax_root'].axhline(color = 'red', linestyle = '--')
            G['ax_root'].set_ylabel('Maximum residual')
        #──────────────────────────────────────────────────────────────┘
        if fit is None:
            if offset == 0: # No skipping of points was possible
                yc.append(y[start - 1])
                if is_debug: #─────────────────────────────────────────┐
                    G['x_plot'] = xc[-2:]
                    G['y_plot'] = yc[-2:]
                #──────────────────────────────────────────────────────┘
            else: # Something weird
                raise RuntimeError('Fit not found')
        else:
            yc.append(fyc(fit, xc[-1]))
            if is_debug: #─────────────────────────────────────────────┐
                G['x_plot'] = G['x'][start -1 + np.arange(- offset, 0)]
                G['y_plot'] = G['fyc'](fit, G['x_plot'])
            #──────────────────────────────────────────────────────────┘
        if is_debug: #─────────────────────────────────────────────────┐
            G['ax_data'].plot(G['x_plot'], G['y_plot'], color = 'red')
        #──────────────────────────────────────────────────────────────┘
        
        limit -= step
        # Setting up to be next estimation
        if limit < offset: offset = limit

        if is_debug: #─────────────────────────────────────────────────┐
            G['start'] = start
            G['ax_data'].plot(xc[-1], yc[-1],'go')
            wait('Next iteration\n')
        #──────────────────────────────────────────────────────────────┘
    else:
        raise StopIteration('Maximum number of iterations reached')
    # Last data point is same as in the unRecord data
    if xc[-2] == xc[-1]: print(xc)
    xc.append(x[-1])
    yc.append(y[-1])
    

    if G['timed']:
        G['runtime'] = time.perf_counter() - G['t_start']

    if is_debug:
        plt.ioff()
    # if xc[-2] == xc[-1]: print(xc)
    return to_ndarray(xc), to_ndarray(yc)
###═════════════════════════════════════════════════════════════════════
### STREAM COMPRESSION
class _StreamRecord(collections.abc.Sized):
    """Class for doing stream compression for data of 1-dimensional
    system of equations
    i.e. single wait variable and one or more output variable
    """
    def __init__(self, x0: float, y0: np.ndarray, tol: np.ndarray, errorfunction: str, use_numba: int, fitset, x2):
        self.is_debug = G['debug']
        if G['timed']: G['t_start'] = time.perf_counter()
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
        self.fit1 = 1
        self.y1 = -self.tol # Initialising
        if self.is_debug: #────────────────────────────────────────────┐
            G.update({'tol': self.tol,
                       'xb': self.xb,
                       'yb': self.yb,
                       'xc': self.xb,
                       'yc': self.yb,
                       'fyc': self.fyc,
                       'limit': self.limit})
            G['fig'], axs = plt.subplots(3,1)
            for ax in axs:
                ax.grid()
            G['ax_data'], G['ax_res'], G['ax_root'] = axs
            G['line_buffer'], = G['ax_data'].plot(0, 0, 'b-',
                                                    label = 'buffer')
            G['line_fit'], = G['ax_data'].plot(0, 0, '-', color = 'orange',
                                                 label = 'fit')

            G['ax_root'].set_ylabel('Tolerance left')

            plt.ion()
            plt.show()
            wait('Record initialised')
        #──────────────────────────────────────────────────────────────┘
    #───────────────────────────────────────────────────────────────────
    def _f2zero(self, i: int) -> tuple:
        '''Function such that i is optimal when f2zero(i) = 0'''

        inds = self.sqrtrange(i)
        residuals, fit = self.f_fit(self.xb[inds], self.yb[inds],
                                    self.xc[-1], self.yc[-1])
        if self.is_debug: #────────────────────────────────────────────┐
            if residuals.shape != (len(inds), len(self.yb[-1])):
                raise ValueError(f'{residuals.shape=}')
            print(f'\t\t{i=}\t{residuals.shape=}')
            print(f'\t\tx {self.xb[inds]} - {self.xb[inds][-1]}')
            indices_all = np.arange(0, i + 1)
            G['ax_data'].plot(self.xb[i], self.yb[i], 'k.')
            G['x_plot'] = self.xb[indices_all]
            G['y_plot'] = self.fyc(fit, G['x_plot'])
            G['line_fit'].set_xdata(G['x_plot'])
            G['line_fit'].set_ydata(G['y_plot'])
            res_all = G['y_plot'] - self.yb[indices_all].flatten()
            G['ax_res'].clear()
            G['ax_res'].grid()
            G['ax_res'].set_ylabel('Residual relative to tolerance')
            G['ax_res'].plot(indices_all, np.abs(res_all) / self.tol -1,
                             'b.', label = 'ignored')
            G['ax_res'].plot(inds, np.abs(residuals) / self.tol - 1,
                             'ro', label = 'sampled')
            G['ax_res'].legend(loc = 'lower right')
            wait('\t\tFitting\n')
        #──────────────────────────────────────────────────────────────┘
        return self.errorfunction(residuals), fit
    #───────────────────────────────────────────────────────────────────
    def squeeze_buffer(self):
        '''Compresses the buffer by one step'''
        #──────────────────────────────────────────────────────────────┘
        offset, fit = interval(self._f2zero, self.x1, self.y1,
                               self.x2, self.y2, self.fit1)
        self.xc.append(self.xb[offset])
        if self.is_debug: #────────────────────────────────────────────┐
            G['ax_root'].clear()
            G['ax_root'].grid()
            G['ax_root'].set_ylabel('Maximum residual')

        if fit is None:
            if offset == 0: # No skipping of points was possible
                self.yc.append(self.yb[offset])
                if self.is_debug: #────────────────────────────────────┐
                    G['x_plot'] = self.xc[-2:]
                    G['y_plot'] = self.yc[-2:]
                #──────────────────────────────────────────────────────┘
            else: # Something weird
                raise RuntimeError('Fit not found')
        else:

            self.yc.append(self.fyc(fit, self.xc[-1]))
            if self.is_debug: #────────────────────────────────────────┐
                G['x_plot'] = self.xb[np.arange(0, offset + 1)]
                G['y_plot'] = G['fyc'](fit, G['x_plot'])
            #──────────────────────────────────────────────────────────┘
        if self.is_debug: #────────────────────────────────────────────┐
            G['ax_data'].plot(G['x_plot'], G['y_plot'], color = 'red')
        #──────────────────────────────────────────────────────────────┘
        self.x1, self.y1, step = 0, self.start_y1, offset + 1

        self.limit -= step
        self._lenb -= step
        self._lenc += 1

        self.x2 = offset # Approximation

        self.xb, self.yb = self.xb[step:], self.yb[step:]
        if self.xc[-1] == self.xb[0]:
            raise IndexError('End of compressed and beginning of buffer are same')
    #───────────────────────────────────────────────────────────────────
    def __call__(self, x_raw, y_raw):
        self.xb.append(x_raw)
        self.yb.append(to_ndarray(y_raw, (-1,)))
        self.limit += 1
        self._lenb += 1

        if self.is_debug: #────────────────────────────────────────────┐
            G['line_buffer'].set_xdata(self.xb)
            G['line_buffer'].set_ydata(self.yb)
        #──────────────────────────────────────────────────────────────┘
        if  self.limit >= self.x2: #───────────────────────────────────┐
            # Converting to numpy arrays for computations
            self.xb = to_ndarray(self.xb)
            self.yb = to_ndarray(self.yb, (self._lenb, -1))

            if self.is_debug: #────────────────────────────────────────┐
                if self.xb.shape != (self._lenb,):
                    raise ValueError(f'xb {self.xb.shape} len {self._lenb}')

                if self.yb.shape != (self._lenb, len(self.yc[0])):
                    raise ValueError(f'{self.yb.shape=}')
            #──────────────────────────────────────────────────────────┘
            self.y2, self.fit2 = self._f2zero(self.x2)

            if self.is_debug: #────────────────────────────────────────┐
                G['xy1'], = G['ax_root'].plot(self.x1, self.y1,'g.')
                G['xy2'], = G['ax_root'].plot(self.x2, self.y2,'b.')
            #──────────────────────────────────────────────────────────┘
            if self.y2 < 0: #──────────────────────────────────────────┐
                if self.is_debug: #────────────────────────────────────┐
                    wait('Calculating new attempt in end\n')
                    G['ax_root'].plot(self.x1, self.y1,'.', color = 'black')
                #──────────────────────────────────────────────────────┘
                self.x1, self.y1, self.fit1 = self.x2, self.y2, self.fit2
                self.x2 *= 2
                self.x2 += 1

                if self.is_debug: #────────────────────────────────────┐
                    print(f'{self.limit=}')
                    print(f'{self.x1=}\t{self.y1=}')
                    print(f'{self.x2=}\t{self.y2=}')
                    G['xy1'].set_xdata(self.x1)
                    G['xy1'].set_ydata(self.y1)
                    G['xy2'].set_xdata(self.x2)
                #──────────────────────────────────────────────────────┘
            else: # Squeezing the buffer
                if self.is_debug: #────────────────────────────────────┐
                    G['xy2'].set_color('red')
                    wait('Points for interval found\n')
                    print(f'{self._lenc=}')
                #──────────────────────────────────────────────────────┘
                self.squeeze_buffer()
            #──────────────────────────────────────────────────────────┘
            # Converting back to lists
            self.xb, self.yb = list(self.xb), list(self.yb)
            if self.is_debug: #────────────────────────────────────────┐
                G['ax_data'].plot(self.xc[-1], self.yc[-1], 'go')
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
        if G['timed']: G['runtime'] = time.perf_counter() - G['t_start']
    #───────────────────────────────────────────────────────────────────
###═════════════════════════════════════════════════════════════════════
class Stream():
    '''Context manager for stream compression of data of
    1 dimensional system of equations'''
    def __init__(self, x0, y0, tol = 1e-2, initial_step = 100,
                 errorfunction = 'maxmaxabs', use_numba = 0, fitset = 'Poly10'):
        self.x0            = x0
        # Variables are columns, e.G. 3xn
        self.y0            = to_ndarray(y0, (-1,))
        self.tol           = to_ndarray(tol, self.y0.shape)

        if isinstance(errorfunction, str): #───────────────────────────┐
            self.errorfunction = get_errorfunction(errorfunction, use_numba, self.tol)
        else:
            self.errorfunction = errorfunction
        #──────────────────────────────────────────────────────────────┘
        if isinstance(fitset, str): #──────────────────────────────────┐
            self.fitset = fitsets[fitset]
        else:
            self.fitset = fitset
        #──────────────────────────────────────────────────────────────┘
        self.use_numba     = use_numba
        self.x2            = initial_step
    #───────────────────────────────────────────────────────────────────
    def __enter__(self):
        self.record = _StreamRecord(self.x0, self.y0, self.tol,
                                          self.errorfunction, self.use_numba, self.fitset, self.x2)
        return self.record
    #───────────────────────────────────────────────────────────────────
    def __exit__(self, exc_type, exc_value, traceback):
        self.record.close()

#%%═════════════════════════════════════════════════════════════════════
def _decompress(x_compressed: np.ndarray, fit_array: np.ndarray, interpolator):
    '''Takes array of fitting parameters and constructs whole function'''
    #───────────────────────────────────────────────────────────────
    def _iteration(x, low = 1):
        index = bisect_left(x_compressed, x, lo = low, hi = fit_array.shape[0]-1)
        return index, interpolator(x, *x_compressed[index-1:(index + 1)],
                                   *fit_array[index-1:(index + 1)])
    #───────────────────────────────────────────────────────────────
    def function(x_input):
        if hasattr(x_input, '__iter__'):
            out = np.full((len(x_input),) + fit_array.shape[1:], np.nan)
            i_c = 1
            for i_out, x in enumerate(x_input):
                i_c, out[i_out] = _iteration(x, i_c)
            return out
        else:
            return _iteration(x_input)[1]
    #───────────────────────────────────────────────────────────────
    return function
#%%═════════════════════════════════════════════════════════════════════
# WRAPPING
# Here are the main external inteface functions
compressors = {'LSQ10': LSQ10}
interpolators = {'Poly10': models.Poly10._interpolate}
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
def decompress(x, y, interpolator = 'Poly10', **kwargs):
    '''Wrapper for easier selection of compression method'''
    if isinstance(interpolator, str):
        try:
            interpolator = interpolators[interpolator]
        except KeyError:
            raise NotImplementedError("Method not in the dictionary of methods")
    return _decompress(x, y, interpolator, **kwargs)
#%%═════════════════════════════════════════════════════════════════════
# HACKS
# A hack to make the package callable
class Pseudomodule(types.ModuleType):
    """Class that wraps the individual plotting functions
    an allows making the module callable"""
    @staticmethod
    def __call__(*args, compressor = 'LSQ10', interpolator = 'Poly10', **kwargs):
        '''Wrapper for easier for combined compression and decompression'''

        return decompress(*compress(*args, compressor = compressor, **kwargs),
                          interpolator = interpolator)

#───────────────────────────────────────────────────────────────────────
fitsets = {'Poly10': models.Poly10}
#%%═════════════════════════════════════════════════════════════════════
# Here the magic happens for making the API module itself also callable
sys.modules[__name__].__class__ = Pseudomodule