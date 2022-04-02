#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numba
import numpy as np
import sys
import time
import types
import collections

import matplotlib.pyplot as plt

from . import reference as ref # Careful with this circular import

# This global dictionary _G is for passing some telemtery and debug arguments
global _G
_G = {}
_G['timed'] = False
_G['debug'] = False
# For Stream compression output
Record = collections.namedtuple('Record', ['x', 'y', 'tol', 'state'],
                                    defaults =  ['closed'])

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
def sqrtrange(start: int, i: int):
    '''~ sqrt(n + 2) equally spaced integers including the i'''
    inds = np.arange(start, i + start + 1, round((i + 1) ** 0.5) )
    inds[-1] = i + start
    return inds
#───────────────────────────────────────────────────────────────────────
def wait(text: str):
    if input(text) in ('e', 'q', 'exit', 'quit'): sys.exit()
#%%═════════════════════════════════════════════════════════════════════
## ERROR TERM
def _maxmaxabs_python(r: np.ndarray, t: np.ndarray) -> float:
    r = np.abs(r)
    # print(f'{r.shape=}')
    # print(f'{t.shape=}')
    m = -1
    for i, k in enumerate(t): # Yes, this is silly
        # n = np.max(r[:,i]) - t[i]
        n = np.max(r[:,i]) - k
        if n > m: m = n
    return m
#───────────────────────────────────────────────────────────────────────
@numba.jit(nopython=True, cache=True)
def _maxmaxabs_numba(r: np.ndarray, t: np.ndarray) -> float:
    r = np.abs(r)
    # print(f'{r.shape=}')
    # print(f'{t.shape=}')
    m = -1
    for i, k in enumerate(t): # Yes, this is silly
        # n = np.max(r[:,i]) - t[i]
        n = np.max(r[:,i]) - k
        if n > m: m = n
    return m
#───────────────────────────────────────────────────────────────────────
def _maxRMS(r: np.ndarray,t: np.ndarray)-> float:
    return np.amax(np.sqrt(np.mean(r * r, axis = 0)) - t)
#───────────────────────────────────────────────────────────────────────
def _maxsumabs(r: np.ndarray,t: np.ndarray) -> float:
    return np.amax(np.sum(np.abs(r) - t, axis = 0) / t)

errorfunctions = {'maxmaxabs': (_maxmaxabs_python, _maxmaxabs_numba),
                  'maxRMS':_maxRMS}
#%%═════════════════════════════════════════════════════════════════════
## ROOT FINDING
def interval(f, x1, y1, x2, y2, fit1):
    '''Returns the last x where f(x)<0'''
    is_debug = _G['debug']
    if is_debug:
        _G['mid'], = _G['ax_root'].plot(x1, y1,'.', color = 'blue')
    while x2 - x1 > 2:
        if is_debug:
            wait('Calculating new attempt in interval\n')
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
            _G['mid'].set_xdata(x_mid)
            _G['mid'].set_ydata(y_mid)
        if y_mid > 0:
            if is_debug:
                wait('Error over tolerance\n')
                _G['ax_root'].plot(x2, y2,'.', color = 'black')
            x2, y2 = x_mid, y_mid
            if is_debug:
                _G['xy2'].set_xdata(x2)
                _G['xy2'].set_ydata(y2)
        else:
            if is_debug:
                wait('Error under tolerance\n')
                _G['ax_root'].plot(x1, y1,'.', color = 'black')
            x1, y1, fit1 = x_mid, y_mid, fit
            if is_debug:
                _G['xy1'].set_xdata(x1)
                _G['xy1'].set_ydata(y1)

    if x2 - x1 == 2: # Points have only one point in between
        if is_debug:
            wait('Points have only one point in between\n')
        y_mid, fit = f(x1+1) # Testing that point
        return (x1+1, fit) if (y_mid <0) else (x1, fit1) # If under, give that fit
    else:
        if is_debug:
            wait('Points have no point in between\n')
        return x1, fit1
#───────────────────────────────────────────────────────────────────────
def droot(f, y1, x2, limit):
    '''Finds the upper limit to interval
    '''
    is_debug = _G['debug']
    x1 = 0
    y2, fit2 = f(x2)
    if is_debug:
        _G['xy1'], = _G['ax_root'].plot(x1, y1,'.', color = 'green')
        _G['xy2'], = _G['ax_root'].plot(x2, y2,'.', color = 'blue')
    fit1 = None
    while y2 < 0:
        if is_debug:
            wait('Calculating new attempt in droot\n')
            _G['ax_root'].plot(x1, y1,'.', color = 'black')
        x1, y1, fit1 = x2, y2, fit2
        x2 *= 2
        x2 += 1
        if is_debug:
            print(f'{limit=}')
            print(f'{x1=}')
            print(f'{y1=}')
            print(f'{x2=}')
            _G['xy1'].set_xdata(x1)
            _G['xy1'].set_ydata(y1)
            _G['xy2'].set_xdata(x2)
        if x2 >= limit:
            if is_debug:
                _G['ax_root'].plot([limit, limit], [y1,0],'.', color = 'blue')
            y2, fit2 = f(limit)
            if y2<0:
                if is_debug:
                    wait('End reached within tolerance\n')
                return limit, fit2
            else:
                if is_debug:
                    wait('End reached outside tolerance\n')
                x2 = limit
                break
        y2, fit2 = f(x2)
        if is_debug:
            print(f'{y2=}')
            _G['xy2'].set_ydata(y2)
    if is_debug:
        _G['xy2'].set_color('red')
        wait('Points for interval found\n')
    return interval(f, x1, y1, x2, y2, fit1)
#───────────────────────────────────────────────────────────────────────
# @numba.jit(nopython=True,cache=True)
def n_lines(x: np.ndarray, y: np.ndarray, x0: float, y0: np.ndarray, tol: float
            ) -> float:
    '''Estimates number of lines required to fit within error tolerance'''

    if (length := len(x)) > 1:
        inds = sqrtrange(0, length - 2) # indices so that x[-1] is not included
        res = (y[-1] - y0) / (x[-1] - x0)*(x[inds] - x0).reshape([-1,1]) - (y[inds] - y0)
        # print(f'sqrtmaxmaxabs {(_maxmaxabs(res, tol)/ tol)** 0.5}')
        # print(f'sqrtmaxsumabs {(_maxsumabs(res, tol)/ tol)** 0.5}')
        # print(f'sqrtmaxRMS {(_maxRMS(res, tol)/ tol)** 0.5}')
        # _maxsumabs(res, tol)
        return 0.5 * (_maxsumabs(res, tol) + 1) ** 0.5 + 1
    else:
        return 1.

###═════════════════════════════════════════════════════════════════════
### BLOCK COMPRESSION
def LSQ10(x: np.ndarray, y: np.ndarray,
          tol = 1e-2, errorfunction = 'maxmaxabs', use_numba = 0) -> tuple:
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
    
    x = to_ndarray(x, (-1, 1))
    y = to_ndarray(y, (len(x), -1))

    tol = to_ndarray(tol, (1, y.shape[1]))

    errf = errorfunctions[errorfunction][use_numba]
    fitset = Poly1
    f_fit = fitset.fit[use_numba]
    fyc = fitset.y_from_fit

    x_c, y_c = [x[0]], [np.array(y[0])]

    # Estimation for the first offset
    offset = round(limit / n_lines(x[start:(end // 2)], y[start:(end // 2)],
                                   x[0], y[0], tol))
    if is_debug:
        _G['x'], _G['y'] = x, y
        _G['fig'], axs = plt.subplots(3,1)
        for ax in axs:
            ax.grid()
        _G['ax_data'], _G['ax_res'], _G['ax_root'] = axs

        _G['ax_data'].fill_between((x).flatten(), (y - tol).flatten(), (y + tol).flatten(), alpha=.3, color = 'blue')

        _G['line_fit'], = _G['ax_data'].plot(0,0,'-',color = 'orange')

        _G['ax_root'].set_ylabel('Tolerance left')

        plt.ion()
        plt.show()
    #───────────────────────────────────────────────────────────────
    def _f2zero(i: int) -> tuple:
        '''Function such that i is optimal when f2zero(i) = 0'''
        # inds = np.arange(start, i + start + 1, round(i**0.5))
        # inds[-1] = i + start
        inds = sqrtrange(start, i)
        residuals, fit = f_fit(x[inds], y[inds], x_c[-1], y_c[-1])
        if is_debug:
            print(f'{residuals.shape=}')
            print(f'x {x[inds][0][0]} - {x[inds][-1][0]}')
            indices_all = np.arange(-1, i + 1) + start
            _G['x_plot'] = _G['x'][indices_all]
            _G['y_plot'] = Poly1.y_from_fit(fit, _G['x_plot'])
            _G['line_fit'].set_xdata(_G['x_plot'])
            _G['line_fit'].set_ydata(_G['y_plot'])
            print(f'{_G["y_plot"].shape=}')
            print(f'{_G["y"][indices_all].shape=}')
            res_all = _G['y_plot'] - _G['y'][indices_all]
            print(f'{res_all.shape=}')
            _G['ax_res'].clear()
            _G['ax_res'].grid()
            _G['ax_res'].set_ylabel('Residual relative to tolerance')
            _G['ax_res'].plot(indices_all - start, np.abs(res_all) / tol -1,
                             '.', color = 'blue', label = 'ignored')
            _G['ax_res'].plot(inds - start, np.abs(residuals) / tol-1,
                             'o', color = 'red', label = 'sampled')
            _G['ax_res'].legend()
            wait('Fitting\n')
        return errf(residuals, tol), fit
    #───────────────────────────────────────────────────────────────
    
    if is_debug:
        wait('Starting\n')
        print(f'{offset=}')
    for _ in range(end): # Prevents infinite loop in case error
        
        offset, fit = droot(_f2zero, -1, offset, limit)
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

        if is_debug:
            _G['ax_data'].plot(x_c[-1], y_c[-1],'.',color = 'green')
            wait('Next iteration\n')
    else:
        raise StopIteration('Maximum number of iterations reached')
    # Last data point is same as in the unRecord data
    x_c.append(x[-1])
    y_c.append(y[-1])

    if _G['timed']:
        _G['runtime'] = time.perf_counter() - _G['t_start']
    
    if is_debug:
        plt.ioff()
    return to_ndarray(x_c, (-1,1)), to_ndarray(y_c)
###═════════════════════════════════════════════════════════════════════
def pick(x, y, tol=1e-2, mins=30, verbosity=0, is_timed=False, use_numba = 0):
    '''Returns inds of data points to select.
    Should be faster than LSQ based'''

    if is_timed: t_start = time.perf_counter()

    zero = 1
    end = len(x)- 1 - zero
    estimate = int(end/n_lines(x,y,x[0],y[0],tol) )+1
    inds = [0]
    errf = errorfunctions['maxmaxabs'][use_numba]

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

        return errf(residuals, tol), None
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
    def __init__(self, x0, y0, tol, errorfunction: str, use_numba: int):
        self.is_debug = _G['debug']
        if _G['timed']: _G['t_start'] = time.perf_counter()
        self.xb, self.yb = [], [] # Buffers for yet-to-be-recorded data
        self.xc, self.yc = [x0], [y0]
        self.start = 0 # Index of starting point for looking for optimum
        self.end = 2 # Index of end point for looking for optimum
        self.tol = tol
        self.state = 'open' # Open means the object is ready to accept more values
        self.errf = errorfunctions[errorfunction][use_numba]
        self.fitset = Poly1
        self.f_fit = self.fitset.fit[use_numba]
        self.fyc = self.fitset.y_from_fit
        self.limit = -1 # Last index of the buffer

        self._lenb = 0 # length of the buffer
        self._lenc = 1 # length of the Record points

        self.tol1 = -self.tol # Initialising
        if self.is_debug: #────────────────────────────────────────────┐
            _G['fig'], axs = plt.subplots(3,1)
            for ax in axs:
                ax.grid()
            _G['ax_data'], _G['ax_res'], _G['ax_root'] = axs
            _G['line_buffer'], = _G['ax_data'].plot(0, 0, '-', color = 'blue',
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

        inds = sqrtrange(0, i)
        residuals, fit = self.f_fit(self.xb[inds], self.yb[inds],
                                    self.xc[-1], self.yc[-1])
        if residuals.shape != (len(inds), len(self.yb[-1])):
                raise ValueError(f'{residuals.shape=}')
        if self.is_debug: #────────────────────────────────────────────┐
            print(f'{residuals.shape=}')
            print(f'x {self.xb[inds][0][0]} - {self.xb[inds][-1][0]}')
            indices_all = np.arange(0, i + 1)
            _G['ax_data'].plot(self.xb[i], self.yb[i], 'k.')
            _G['x_plot'] = self.xb[indices_all]
            _G['y_plot'] = Poly1.y_from_fit(fit, _G['x_plot'])
            _G['line_fit'].set_xdata(_G['x_plot'])
            _G['line_fit'].set_ydata(_G['y_plot'])
            res_all = _G['y_plot'] - self.yb[indices_all].reshape(-1,1)
            _G['ax_res'].clear()
            _G['ax_res'].grid()
            _G['ax_res'].set_ylabel('Residual relative to tolerance')
            _G['ax_res'].plot(indices_all, np.abs(res_all) / self.tol -1,
                             '.', color = 'blue', label = 'ignored')
            _G['ax_res'].plot(inds, np.abs(residuals) / self.tol - 1,
                             'o', color = 'red', label = 'sampled')
            _G['ax_res'].legend()
            wait('Fitting\n')
        #──────────────────────────────────────────────────────────────┘
        return self.errf(residuals, self.tol), fit
    #───────────────────────────────────────────────────────────────────
    def squeeze_buffer(self):
        '''Compresses the buffer by one step'''
        offset, fit = interval(self._f2zero, self.start, self.tol1,
                               self.limit, self.tol2, self.fit1)
        if fit is None: raise RuntimeError('Fit not found')
        if self.is_debug: #────────────────────────────────────────────┐
            # print(f'err {self.errf(self.fyc(fit, x[self.start + offset]) - y[self.start + offset], self.tol)}')
            print(f'{self.start=}\t{offset=}\t{self.end=}\t')
            print(f'{fit=}')
            _G['ax_root'].clear()
            _G['ax_root'].grid()
            _G['ax_root'].set_ylabel('Maximum residual')
            _G['ax_data'].plot(_G['x_plot'], _G['y_plot'], color = 'red')
        #──────────────────────────────────────────────────────────────┘
        self.start, self.tol1, step = 0, - self.tol, offset + 1
        
        self.xc.append(self.xb[offset])
        self.yc.append(self.fyc(fit, self.xc[-1]))

        self.limit -= step
        
        self._lenb -= step
        self._lenc += 1

        self.end = min(self.limit, offset) # Approximation

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
        if  self.limit >= self.end:
            # Converting to numpy arrays for computations
            self.xb = to_ndarray(self.xb, (self._lenb, 1))
            self.yb = to_ndarray(self.yb, (self._lenb, -1))

            if self.xb.shape != (self._lenb, 1):
                raise ValueError(f'{self.xb.shape=}')

            if self.yb.shape != (self._lenb, len(self.yc[0])):
                raise ValueError(f'{self.yb.shape=}')
            
            self.tol2, self.fit2 = self._f2zero(self.limit)

            if self.is_debug: #────────────────────────────────────────┐
                _G['xy1'], = _G['ax_root'].plot(self.start, self.tol1,'.', color = 'green')
                _G['xy2'], = _G['ax_root'].plot(self.end, self.tol2,'.', color = 'blue')
            #──────────────────────────────────────────────────────────┘
            if self.tol2 < 0:
                if self.is_debug:
                    wait('Calculating new attempt in end\n')
                    _G['ax_root'].plot(self.start, self.tol1,'.', color = 'black')
                self.start, self.tol1, self.fit1 = self.end, self.tol2, self.fit2
                self.end *= 2
                self.end += 1

                if self.is_debug: #────────────────────────────────────┐
                    print(f'{self.limit=}')
                    print(f'{self.start=}')
                    print(f'{self.tol1=}')
                    print(f'{self.end=}')
                    _G['xy1'].set_xdata(self.start)
                    _G['xy1'].set_ydata(self.tol1)
                    _G['xy2'].set_xdata(self.end)
                #──────────────────────────────────────────────────────┘
            else: # Squeezing the buffer
                self.squeeze_buffer()
            if self.is_debug: #────────────────────────────────────────┐
                _G['ax_data'].plot(self.xc[-1], self.yc[-1], '.', color = 'green')
                wait('Next iteration\n')
            #──────────────────────────────────────────────────────────┘
            # Converting back to lists
            self.xb, self.yb = list(self.xb.flatten()), list(self.yb)
            if self.yb[-1].shape != (1,):
                raise ValueError(f'{self.yb[-1].shape=}')
            if self.yc[-1].shape != (1,):
                raise ValueError(f'{self.yc[-1].shape=}')
            # if self.yb[0].shape != (1,):
            #     raise ValueError(f'{self.yb[0].shape=}')
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
        self.xb = to_ndarray(self.xb, (self._lenb, 1))
        self.yb = to_ndarray(self.yb, (self._lenb, -1))
        # print(f'Calling f2zero with {self.limit=}')
        self.tol2, self.fit2 = self._f2zero(self.limit)

        while self.tol2 > 0:
            self.squeeze_buffer()
            self.tol2, self.fit2 = self._f2zero(self.limit)
        
        self.xc.append(to_ndarray(self.xb[-1], (1,)))
        self.yc.append(to_ndarray(self.yb[-1], (1,)))
        # Deleting unnecessary attributes
        
        if self.is_debug: plt.ioff()
        
        # Final packing and cleaning
        self.x, self.y = to_ndarray(self.xc), to_ndarray(self.yc)
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
    def __init__(self, x0, y0, tol = 1e-2,
                 errorfunction = 'maxmaxabs', use_numba = 0):
        self.x0            = to_ndarray(x0, (1,))
        # Variables are columns, e._G. 3xn
        self.y0            = to_ndarray(y0, (-1,))
        self.tol           = to_ndarray(tol, self.y0.shape)
        self.errorfunction = errorfunction
        self.use_numba     = use_numba
    #───────────────────────────────────────────────────────────────────
    def __enter__(self):
        self.record = _StreamRecord(self.x0, self.y0, self.tol,
                                          self.errorfunction, self.use_numba)
        return self.record
    #───────────────────────────────────────────────────────────────────
    def __exit__(self, exc_type, exc_value, traceback):
        self.record.close()
        # self.record = Record(self.record.xc, self.record.yc, self.record.tol)
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
    def fit_python(x: np.ndarray, y: np.ndarray, x0, y0: np.ndarray) -> tuple:
        '''Takes block of data, previous fitting parameters and calculates next fitting parameters'''

        Dx = x - x0
        Dy = y - y0
        a = (Dx.T @ Dy / Dx.T.dot(Dx)).flatten()
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
    @numba.jit(nopython=True, cache=True)
    def fit_numba(x: np.ndarray, y: np.ndarray, x0, y0: np.ndarray) -> tuple:
        '''Takes block of data, previous fitting parameters and calculates next fitting parameters'''

        Dx = x - x0
        Dy = y - y0
        a = Dx.T @ Dy / Dx.T.dot(Dx).flatten()
        b = y0 - a * x0
        # print(f'{a.shape=}')
        # print(f'{Dx.shape=}')
        # print(f'{y.shape=}')
        # print(f'{y0.shape=}')
        # print(f'{Dy.shape=}')
        return (a * Dx - Dy, (a,  b))
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    def y_from_fit(fit: tuple, x: np.ndarray) -> np.ndarray:
        '''Converts the fitting parameters and x to storable y values'''
        return fit[0]*x + fit[1]
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
    #───────────────────────────────────────────────────────────────────
    fit = (fit_python, fit_numba)
#%%═════════════════════════════════════════════════════════════════════
# HACKS
# A hack to make the package callable
class Pseudomodule(types.ModuleType):
    '''Class that wraps the individual plotting functions
    an allows making the module callable'''
    @staticmethod
    def __call__(*args, method='LSQ10', **kwargs):
        '''Wrapper for easier selection of compression method'''
        try:
            compressor = methods[method]
        except KeyError:
            raise NotImplementedError("Method not in the dictionary of methods")
        return compressor(*args, **kwargs)
#───────────────────────────────────────────────────────────────────────
# Here the magic happens for making the API module itself also callable
sys.modules[__name__].__class__ = Pseudomodule