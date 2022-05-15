#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
API
========================================================================

Connection point for all package utilities provided
'''
import collections
import numpy as np
import sys
import time
import types
from bisect import bisect_left
import matplotlib.pyplot as plt

from . import f2zero
from . import GLOBALS
from . import auxiliaries as aux
from .auxiliaries import to_ndarray, wait, sqrtranges
from .root import droot, droot_debug, interval, interval_debug
from . import errorfunctions
from . import models
from . import reference as ref # Careful with this circular import
# This global dictionary G is for passing some telemtery and debug arguments
global G
G = GLOBALS.dictionary

fitsets = {'Poly10': models.Poly10,
           'Poly1100': models.Poly1100}

# @numba.jit(nopython=True,cache=True)
def n_lines(x: np.ndarray, y: np.ndarray, x0: float, y0: np.ndarray, tol: float
            ) -> float:
    '''Estimates number of lines required to fit within error tolerance'''

    if (length := len(x)) > 1:
        inds = sqrtranges[0](length - 2) # indices so that x[-1] is not included
        res = (y[-1] - y0) / (x[-1] - x0)*(x[inds] - x0).reshape([-1, 1]) - (y[inds] - y0)

        reference = errorfunctions.maxsumabs(res, tol)
        if reference < 0: reference = 0
        return 0.5 * reference ** 0.5 + 1
    else:
        return 1.
#%%═════════════════════════════════════════════════════════════════════
# BLOCK COMPRESSION
def LSQ10(x_in: np.ndarray, y_in: np.ndarray, tol = 1e-2, initial_step = None,
          errorfunction = 'maxmaxabs', use_numba = 0, fitset = 'Poly10') -> tuple:
    '''Compresses the data of 1-dimensional system of equations
    i.e. single wait variable and one or more output variable
    '''
    is_debug = G['debug']
    if G['timed']:  G['t_start'] = time.perf_counter()
    start   = 1 # Index of starting point for looking for optimum
    end     = len(x_in) - 1 # Number of datapoints -1, i.e. the last index
    limit   = end - start

    x       = to_ndarray(x_in)
    y       = to_ndarray(y_in, (len(x), -1))
    tol     = to_ndarray(tol, y[0].shape)
    xc, yc = [x[0]], [y[0]]
    start_y1 = - np.amax(tol) # Starting value for discrete root calculation

    sqrtrange = sqrtranges[use_numba]
    if isinstance(errorfunction, str):
        errorfunction = errorfunctions.get(errorfunction, use_numba)
    if isinstance(fitset, str):
        fitset = fitsets[fitset]
    f_fit = fitset.fit[use_numba]

    # Estimation for the first offset
    if initial_step is None:
        mid = end // 2
        offset = round(limit / n_lines(x[1:mid], y[1:mid], x[0], y[0], tol))
    else:
        offset = initial_step
    
    if is_debug: G.update(aux.debugsetup(x, y, tol, fitset, start))
    #───────────────────────────────────────────────────────────────────
    for _ in range(end): # Prevents infinite loop in case error
        if x[start-1] != xc[-1]:
            raise IndexError(f'Indices out of sync {start}')
        if is_debug:
            offset, fit = droot_debug(f2zero.get_debug(x[start:], y[start:],
                                                       xc[-1], yc[-1],
                                                       tol, sqrtrange,
                                                       f_fit, errorfunction),
                                      start_y1, offset, limit)
        else:
            offset, fit = droot(f2zero.get(x[start:], y[start:],
                                           xc[-1], yc[-1],
                                           tol, sqrtrange,
                                           f_fit, errorfunction),
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
            yc.append(fit)
            if is_debug: #─────────────────────────────────────────────┐
                G['x_plot'] = G['x'][start -1 + np.arange(- offset, 0)]
                G['y_plot'] = G['interp'](G['x_plot'], *xc[-2:], *yc[-2:])
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
    xc.append(x[-1])
    yc.append(fit)

    if G['timed']:
        G['runtime'] = time.perf_counter() - G['t_start']

    if is_debug:
        plt.ioff()
    # if xc[-2] == xc[-1]: print(xc)
    return to_ndarray(xc), to_ndarray(yc)
#%%═════════════════════════════════════════════════════════════════════
# STREAM COMPRESSION
class _StreamRecord(collections.abc.Sized):
    """Class for doing stream compression for data of 1-dimensional
    system of equations
    i.e. single wait variable and one or more output variable
    """
    def __init__(self, x0: float, y0: np.ndarray, tol: np.ndarray, errorfunction: str, f_fit, sqrtrange, x2):
        if G['timed']: G['t_start'] = time.perf_counter()
        self.xb, self.yb = [], [] # Buffers for yet-to-be-recorded data
        self.xc, self.yc = [x0], [y0]
        self.x1         = 0 # Index of starting point for looking for optimum
        self.x2         = x2
        self.tol        = tol
        self.start_y1   = -np.amax(tol) # Default starting value
        self.state      = 'open' # The object is ready to accept more values
        self.errorfunction = errorfunction
        self.f_fit      = f_fit 
        self.sqrtrange  = sqrtrange
        self.limit      = -1 # Last index of the buffer
        self._lenb      = 0 # length of the buffer
        self._lenc      = 1 # length of the Record points
        self.fit1       = np.array((1))
        self.y1         = -self.tol # Initialising
    # #───────────────────────────────────────────────────────────────────
    # def _f2zero(self, i: int) -> tuple:
    #     '''Function such that i is optimal when f2zero(i) = 0'''

    #     inds = self.sqrtrange(i)
    #     residuals, fit = self.f_fit(self.xb[inds], self.yb[inds],
    #                                 self.xc[-1], self.yc[-1])
    #     return self.errorfunction(residuals, self.tol), fit
    #───────────────────────────────────────────────────────────────────
    def squeeze_buffer(self, x1, y1, x2, y2):
        '''Compresses the buffer by one step'''
        #──────────────────────────────────────────────────────────────┘
        offset, fit = interval(self.f2zero, x1, y1, x2, y2, self.fit1)
        self.xc.append(self.xb[offset])
        if fit:
            self.yc.append(fit)
        else:
            if offset == 0: # No skipping of points was possible
                self.yc.append(self.yb[offset])
            else: # Something weird
                raise RuntimeError('Fit not found')
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
        if  self.limit >= self.x2: #───────────────────────────────────┐
            # Converting to numpy arrays for computations
            self.xb = to_ndarray(self.xb)
            self.yb = to_ndarray(self.yb, (self._lenb, -1))
            self.f2zero = f2zero.get(self.xb, self.yb, self.xc[-1], self.yc[-1],
                                     self.tol, self.sqrtrange,
                                     self.f_fit, self.errorfunction)
            self.y2, self.fit2 = self.f2zero(self.x2)

            if self.y2 < 0: #──────────────────────────────────────────┐
                self.x1, self.y1, self.fit1 = self.x2, self.y2, self.fit2
                self.x2 *= 2
                self.x2 += 1
            else: # Squeezing the buffer
                self.squeeze_buffer(self.x1, self.y1, self.x2, self.y2)
            #──────────────────────────────────────────────────────────┘
            # Converting back to lists
            self.xb, self.yb = list(self.xb), list(self.yb)
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
        '''Colses the context manager'''
        self.state = 'closing'
        # Converting to numpy arrays for computations
        self.xb = to_ndarray(self.xb)
        self.yb = to_ndarray(self.yb, (self._lenb, -1))
        self.f2zero = f2zero.get(self.xb, self.yb, self.xc[-1], self.yc[-1],
                                     self.tol, self.sqrtrange,
                                     self.f_fit, self.errorfunction)
        if self.x2 > self.limit: self.x2 = self.limit

        self.y2, self.fit2 = self.f2zero(self.x2)

        while self.y2 > 0: #───────────────────────────────────────────┐
            self.squeeze_buffer(self.x1, self.y1, self.x2, self.y2)
            
            if self.x2 > self.limit: self.x2 = self.limit
            self.f2zero = f2zero.get(self.xb, self.yb, self.xc[-1], self.yc[-1],
                                     self.tol, self.sqrtrange,
                                     self.f_fit, self.errorfunction)
            self.y2, self.fit2 = self.f2zero(self.x2)
        #──────────────────────────────────────────────────────────────┘
        self.xc.append(self.xb[-1])
        self.yc.append(to_ndarray(self.yb[-1], (1,)))

        # Final packing and cleaning
        self.x = to_ndarray(self.xc, (self._lenc + 1,))
        self.y = to_ndarray(self.yc, (self._lenc + 1, -1))
        for key in tuple(self.__dict__):
            if key not in {'x', 'y', 'state', 'tol'}:
                del self.__dict__[key]

        self.state = 'closed'
        if G['timed']: G['runtime'] = time.perf_counter() - G['t_start']
    #───────────────────────────────────────────────────────────────────
class _StreamRecord_debug(collections.abc.Sized):
    """Class for doing stream compression for data of 1-dimensional
    system of equations
    i.e. single wait variable and one or more output variable
    """
    def __init__(self, x0: float, y0: np.ndarray, tol: np.ndarray, errorfunction: str, f_fit, sqrtrange, x2, interpolator):
        if G['timed']: G['t_start'] = time.perf_counter()
        self.xb, self.yb = [], [] # Buffers for yet-to-be-recorded data
        self.xc, self.yc = [x0], [y0]
        self.x1         = 0 # Index of starting point for looking for optimum
        self.x2         = x2
        self.tol        = tol
        self.start_y1   = -np.amax(tol) # Default starting value
        self.state      = 'open' # The object is ready to accept more values
        self.errorfunction = errorfunction
        self.f_fit      = f_fit 
        self.sqrtrange  = sqrtrange
        self.limit      = -1 # Last index of the buffer

        self._lenb      = 0 # length of the buffer
        self._lenc      = 1 # length of the Record points
        self.fit1       = 1
        self.y1         = -self.tol # Initialising

        G.update({'tol': self.tol,
                    'x': np.array(self.xb),
                    'y': np.array(self.yb),
                    'xc': self.xb,
                    'yc': self.yb,
                    'interp': interpolator,
                    'limit': self.limit,
                    'start': 1})
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
        wait('Initialised')
    #───────────────────────────────────────────────────────────────────
    def squeeze_buffer(self, x1, y1, x2, y2):
        '''Compresses the buffer by one step'''

        offset, fit = interval_debug(self.f2zero, x1, y1, x2, y2, self.fit1)
        self.xc.append(self.xb[offset])

        G['ax_root'].clear()
        G['ax_root'].grid()
        G['ax_root'].set_ylabel('Maximum residual')

        if fit:
            self.yc.append(fit)
        else:
            if offset == 0: # No skipping of points was possible
                self.yc.append(self.yb[offset])
                G['x_plot'] = self.xc[-2:]
                G['y_plot'] = self.yc[-2:]
            else: # Something weird
                raise RuntimeError('Fit not found')

            G['x_plot'] = self.xb[np.arange(0, offset + 1)]
            G['y_plot'] = G['interp'](G['x_plot'], *self.xc[-2:], *self.yc[-2:])

        G['ax_data'].plot(G['x_plot'], G['y_plot'], color = 'red')

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

        G['line_buffer'].set_xdata(self.xb)
        G['line_buffer'].set_ydata(self.yb)

        if  self.limit >= self.x2: #───────────────────────────────────┐
            # Converting to numpy arrays for computations
            self.xb = to_ndarray(self.xb)
            self.yb = to_ndarray(self.yb, (self._lenb, -1))
            G['x'] = self.xb
            G['y'] = self.yb
            self.f2zero = f2zero.get_debug(self.xb, self.yb,
                                           self.xc[-1], self.yc[-1],
                                           self.tol, self.sqrtrange,
                                           self.f_fit, self.errorfunction)
            if self.xb.shape != (self._lenb,):
                raise ValueError(f'xb {self.xb.shape} len {self._lenb}')

            if self.yb.shape != (self._lenb, len(self.yc[0])):
                raise ValueError(f'{self.yb.shape=}')

            self.y2, self.fit2 = self.f2zero(self.x2)

            G['xy1'], = G['ax_root'].plot(self.x1, self.y1,'g.')
            G['xy2'], = G['ax_root'].plot(self.x2, self.y2,'b.')

            if self.y2 < 0: #──────────────────────────────────────────┐

                wait('Calculating new attempt in end\n')
                G['ax_root'].plot(self.x1, self.y1,'.', color = 'black')

                self.x1, self.y1, self.fit1 = self.x2, self.y2, self.fit2
                self.x2 *= 2
                self.x2 += 1

                print(f'{self.limit=}')
                print(f'{self.x1=}\t{self.y1=}')
                print(f'{self.x2=}\t{self.y2=}')
                G['xy1'].set_xdata(self.x1)
                G['xy1'].set_ydata(self.y1)
                G['xy2'].set_xdata(self.x2)

            else: # Squeezing the buffer
                G['xy2'].set_color('red')
                wait('Points for interval found\n')
                print(f'{self._lenc=}')

                self.squeeze_buffer(self.x1, self.y1, self.x2, self.y2)
            #──────────────────────────────────────────────────────────┘
            # Converting back to lists
            self.xb, self.yb = list(self.xb), list(self.yb)

            G['ax_data'].plot(self.xc[-1], self.yc[-1], 'go')
            wait('Next iteration\n')
            if self.yb[-1].shape != (1,):
                raise ValueError(f'{self.yb[-1].shape=}')
            if self.yc[-1].shape != (1,):
                raise ValueError(f'{self.yc[-1].shape=}')
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
        self.f2zero = f2zero.get_debug(self.xb, self.yb,
                                           self.xc[-1], self.yc[-1],
                                           self.tol, self.sqrtrange,
                                           self.f_fit, self.errorfunction)
        self.x2 = min(self.x2, self.limit)
        self.y2, self.fit2 = self.f2zero(self.x2)

        while self.y2 > 0: #───────────────────────────────────────────┐
            self.squeeze_buffer(self.x1, self.y1, self.x2, self.y2)
            self.f2zero = f2zero.get_debug(self.xb, self.yb,
                                           self.xc[-1], self.yc[-1],
                                           self.tol, self.sqrtrange,
                                           self.f_fit, self.errorfunction)
            self.x2 = min(self.x2, self.limit)
            self.y2, self.fit2 = self.f2zero(self.x2)
        #──────────────────────────────────────────────────────────────┘
        self.xc.append(self.xb[-1])
        self.yc.append(to_ndarray(self.yb[-1], (1,)))

        plt.ioff()
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
            self.errorfunction = errorfunctions.get(errorfunction, use_numba)
        else:
            self.errorfunction = errorfunction
        #──────────────────────────────────────────────────────────────┘
        if isinstance(fitset, str): #──────────────────────────────────┐
            self.fitset = fitsets[fitset]
        else:
            self.fitset = fitset
        #──────────────────────────────────────────────────────────────┘
        self.f_fit      = self.fitset.fit[use_numba]
        self.sqrtrange  = sqrtranges[use_numba]
        self.use_numba     = use_numba
        self.x2            = initial_step
    #───────────────────────────────────────────────────────────────────
    def __enter__(self):
        if G['debug']:
            self.record = _StreamRecord_debug(self.x0, self.y0, self.tol,
                                          self.errorfunction, self.f_fit, self.sqrtrange, self.x2, self.fitset.interpolate)
        else:
            self.record = _StreamRecord(self.x0, self.y0, self.tol,
                                          self.errorfunction, self.f_fit, self.sqrtrange, self.x2)
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
#%%═════════════════════════════════════════════════════════════════════
# Here the magic happens for making the API module itself also callable
sys.modules[__name__].__class__ = Pseudomodule