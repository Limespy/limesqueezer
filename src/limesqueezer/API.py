#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
API
========================================================================

Connection point for all package utilities provided
'''
from bisect import bisect_left
import collections
from matplotlib import pyplot as plt
import numpy as np
import sys
import time
import types
from typing import Callable

from .auxiliaries import to_ndarray, wait, sqrtranges, debugsetup, stats
from . import errorfunctions
from . import f2zero
from .GLOBALS import G
from . import models
from . import reference as ref # Careful with this circular import
from .root import droot, droot_debug, interval, interval_debug

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
def LSQ10(x_in: np.ndarray,
          y_in: np.ndarray, /,
          tolerances: float | tuple = (1e-2, 1e-2, 0),
          initial_step: int | None  = None,
          errorfunction:
            str | Callable[[np.ndarray, np.ndarray, tuple[float, float, float]], float]= 'maxmaxabs',
          use_numba: int  = 0,
          fitset: str     = 'Poly10',
          keepshape: bool = False
          ) -> tuple[np.ndarray, np.ndarray]:
    '''Compresses the data of 1-dimensional system of equations
    i.e. single wait variable and one or more output variable

    Parameters
    ----------
    x_in : np.ndarray
        x-coordiantes of the points to be compressed
    y_in : np.ndarray
        y-coordinates of the points to be compressed
    tolerances : tuple, default (1e-2, 1e-2, 0)
        tolerances, Falloff determines how much the absolute error is
        reuduced as y value grows.
            If 3 values: (relative, absolute, falloff)
            If 1 values: (relative, absolute, 0)
            If 1 value:  (0, absolute, 0)
    initial_step : int, default None
        First compression step to be calculated. If None, it is automatically calculated. Provide for testing purposes or for miniscule reduction in setup time
    errorfunction : str, default 'maxmaxabs'
        Function which is used to compute the error of the fit, by 
    use_numba : int, default 0
        For using functions with Numba JIT compilation set as 1
    fitset : str, default 'Poly10'
        Name of the fitting function set
    keepshape : bool, default False
        Whether the output is in similar shape to input

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Compressed  x and Y as numpy ndarrays

    Raises
    ------
    IndexError
        Compressor has internally gotten out of sync due to internal errors.
    RuntimeError
        Creating a fit failed for some unknown reason and returned None.
    StopIteration
        For some reason the compression reached the maximum number of
        iterations possible. Either the input is flawed or compression has
        errors.
    '''
    # Initialisation for main loop
    is_debug = G['debug']
    if G['timed']:  G['t_start'] = time.perf_counter()
    xlen    = len(x_in)
    x       = to_ndarray(x_in)
    y       = to_ndarray(y_in, (xlen, -1))
    if isinstance(tolerances, (int, float)):
        tolerances = (0, tolerances, 0)
    elif hasattr(tolerances, '__iter__'):
        len_tol = len(tolerances)
        if not (0 < len_tol < 4):
            raise ValueError(f'Tolerances length should be 1-3, was {len_tol}')
        elif len(tolerances) == 1:
            tolerances = (0, *tolerances, 0)
        elif len(tolerances) == 2:
            tolerances = (*tolerances, 0)
    else:
        raise TypeError(
            'Tolerances should be a number or sequence of 1-3 numbers')
    # Relative, absolute, falloff
    tol     = tuple([to_ndarray(tol, y[0].shape) for tol in tolerances])

    xc, yc = [x[0]], [y[0]]

    start_err1 = - np.amax(tol[1]) # Starting value for discrete root calculation

    start   = 1 # Index of starting point for looking for optimum
    end     = xlen - 1 # Number of datapoints -1, i.e. the last index
    limit   = end - start

    sqrtrange = sqrtranges[use_numba]
    if isinstance(errorfunction, str):
        errorfunction = errorfunctions.get(errorfunction, use_numba)
    if isinstance(fitset, str):
        fitset = models.get(fitset)
    f_fit = fitset.fit[use_numba]

    # Estimation for the first offset
    if initial_step:
        offset = initial_step
    else:
        mid = end // 2
        offset = round(limit / n_lines(x[1:mid], y[1:mid], x[0], y[0], tol[1]))

    if is_debug:
        G.update(debugsetup(x, y, tol[1], fitset, start))
        get_f2z = f2zero.get
        solver = droot_debug
    else:
        get_f2z = f2zero.get
        solver = droot
    #───────────────────────────────────────────────────────────────────
    # Main loop
    for _ in range(end): # Prevents infinite loop in case error
        if x[start-1] != xc[-1]:
            raise IndexError(f'Indices out of sync {start}')
        offset, fit = solver(get_f2z(x[start:], y[start:], xc[-1], yc[-1],
                                     tol, sqrtrange, f_fit, errorfunction),
                             start_err1, offset, limit)
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
                raise RuntimeError('Fit returned was None, check fit functions for errors')
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
    # Finalising
    xc.append(x[-1])
    yc.append(y[-1])

    if G['timed']: G['runtime'] = time.perf_counter() - G['t_start']

    if is_debug: plt.ioff()

    if keepshape: # returning in same shape as it came in
        yshape = tuple([len(xc) if l == xlen else l for l in y_in.shape])
        xshape = tuple([len(xc) if l == xlen else l for l in x_in.shape])
        return to_ndarray(xc, xshape), to_ndarray(yc, yshape)
    else:
        return np.array(xc), np.array(yc)
#%%═════════════════════════════════════════════════════════════════════
# STREAM COMPRESSION
class _StreamRecord(collections.abc.Sized):
    """Class for doing stream compression for data of 1-dimensional
    system of equations
    i.e. single wait variable and one or more output variable
    """
    def __init__(self,
                 x0: float,
                 y0: np.ndarray,
                 x_type: type,
                 y_type: type,
                 tol: np.ndarray,
                 errorfunction: str,
                 f_fit: Callable,
                 sqrtrange: Callable,
                 x2: int,
                 get_f2zero: Callable):
        if G['timed']: G['t_start'] = time.perf_counter()
        self.xb, self.yb = [], [] # Buffers for yet-to-be-recorded data
        self.xc, self.yc = [x0], [y0]
        self.x_type, self.y_type = x_type, y_type
        self.x1         = 0 # Index of starting point for looking for optimum
        self.x2         = x2
        self.tol        = tol
        self.start_err1   = -np.amax(tol[1]) # Default starting value
        self.state      = 'open' # The object is ready to accept more values
        self.errorfunction = errorfunction
        self.f_fit      = f_fit 
        self.sqrtrange  = sqrtrange
        self.limit      = -1 # Last index of the buffer
        self._lenc      = 1 # length of the Record points
        self.fit1       = np.array((1)) 
        self.err1         = self.start_err1 # Initialising
        self.get_f2zero = get_f2zero
    #───────────────────────────────────────────────────────────────────
    def _update_f2zero(self):
        self.f2zero = self.get_f2zero(self.xb, self.yb,
                                          self.xc[-1], self.yc[-1],
                                          self.tol, self.sqrtrange,
                                          self.f_fit, self.errorfunction)
        self.err2, self.fit2 = self.f2zero(self.x2)
    #───────────────────────────────────────────────────────────────────
    def squeeze_buffer(self, x1: float, err1: np.ndarray,
                       x2: float, err2: np.ndarray):
        '''Compresses the buffer by one step'''
        #──────────────────────────────────────────────────────────────┘
        offset, fit = interval(self.f2zero, x1, err1, x2, err2, self.fit1)
        self.xc.append(self.xb[offset])
        if fit is None:
            if offset == 0: # No skipping of points was possible
                self.yc.append(self.yb[offset])
            else: # Something weird
                raise RuntimeError('Fit not found')
        else:
            self.yc.append(fit)
        self.x1, self.err1, step = 0, self.start_err1, offset + 1

        self.limit -= step
        self._lenc += 1

        self.x2 = offset # Approximation

        self.xb, self.yb = self.xb[step:], self.yb[step:]

        if self.xc[-1] == self.xb[0]:
            raise IndexError('End of compressed and beginning of buffer are same')
    #───────────────────────────────────────────────────────────────────
    def __call__(self, x_raw, y_raw) -> bool:
        _squeezed = False # For tracking if the buffer was compressed

        if type(x_raw) != self.x_type:
            raise TypeError('Type of the x not same as the initial value')
        if type(y_raw) != self.y_type:
            raise TypeError('Type of the y not same as the initial value')

        self.xb.append(x_raw)
        self.yb.append(to_ndarray(y_raw, (-1,)))
        self.limit += 1
        if  self.limit >= self.x2: #───────────────────────────────────┐
            # Converting to numpy arrays for fitting
            self.xb = to_ndarray(self.xb)
            self.yb = to_ndarray(self.yb, (self.limit + 1, -1))
            self._update_f2zero()

            if self.err2 < 0: #──────────────────────────────────────────┐
                self.x1, self.err1, self.fit1 = self.x2, self.err2, self.fit2
                self.x2 *= 2
                self.x2 += 1
            else: # Squeezing the buffer
                self.squeeze_buffer(self.x1, self.err1, self.x2, self.err2)
                _squeezed = True
            #──────────────────────────────────────────────────────────┘
            # Converting back to lists
            self.xb, self.yb = list(self.xb), list(self.yb)
        #──────────────────────────────────────────────────────────────┘
        return _squeezed
    #───────────────────────────────────────────────────────────────────
    def __len__(self):
        return self._lenc + self.limit + 1
    #───────────────────────────────────────────────────────────────────
    def __str__(self):
        return f'{self.x=} {self.y=} {self.tol=}'
    #───────────────────────────────────────────────────────────────────
    def close(self):
        '''Closes the context manager'''
        self.state = 'closing'
        # Clamping x2 to not go over the last buffer index
        if self.x2 > self.limit: self.x2 = self.limit
        # Converting to numpy arrays for fitting
        self.xb = to_ndarray(self.xb)
        self.yb = to_ndarray(self.yb, (self.limit + 1, -1))
        self._update_f2zero()
        while self.err2 > 0: #─────────────────────────────────────────┐
            self.squeeze_buffer(self.x1, self.err1, self.x2, self.err2)
            # Clamping x2 to not go over the last buffer index
            if self.x2 > self.limit: self.x2 = self.limit
            self._update_f2zero()
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
    def __init__(self,
                 x0: float,
                 y0: np.ndarray,
                 x_type: type,
                 y_type: type,
                 tol: np.ndarray,
                 errorfunction: str,
                 f_fit: Callable,
                 sqrtrange: Callable,
                 x2: int,
                 get_f2zero: Callable,
                 interpolator):
        if G['timed']: G['t_start'] = time.perf_counter()
        self.xb, self.yb = [], [] # Buffers for yet-to-be-recorded data
        self.xc, self.yc = [x0], [y0]
        self.x_type, self.y_type = x_type, y_type
        self.x1         = 0 # Index of starting point for looking for optimum
        self.x2         = x2
        self.tol        = tol
        self.start_err1   = -np.amax(tol[1]) # Default starting value
        self.state      = 'open' # The object is ready to accept more values
        self.errorfunction = errorfunction
        self.f_fit      = f_fit 
        self.sqrtrange  = sqrtrange
        self.limit      = -1 # Last index of the buffer

        self._lenc      = 1 # length of the Record points
        self.fit1       = 1
        self.err1         = self.start_err1 # Initialising

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
    def squeeze_buffer(self, x1, err1, x2, err2):
        '''Compresses the buffer by one step'''

        offset, fit = interval_debug(self.f2zero, x1, err1, x2, err2, self.fit1)
        self.xc.append(self.xb[offset])

        G['ax_root'].clear()
        G['ax_root'].grid()
        G['ax_root'].set_ylabel('Maximum residual')

        if fit is None:
            if offset == 0: # No skipping of points was possible
                self.yc.append(self.yb[offset])
                G['x_plot'] = self.xc[-2:]
                G['y_plot'] = self.yc[-2:]
            else: # Something weird
                raise RuntimeError('Fit not found')
        else:
            self.yc.append(fit)

            G['x_plot'] = self.xb[np.arange(0, offset + 1)]
            G['y_plot'] = G['interp'](G['x_plot'], *self.xc[-2:], *self.yc[-2:])

        G['ax_data'].plot(G['x_plot'], G['y_plot'], color = 'red')

        self.x1, self.err1, step = 0, self.start_err1, offset + 1

        self.limit -= step
        self._lenc += 1

        self.x2 = offset # Approximation

        self.xb, self.yb = self.xb[step:], self.yb[step:]
        if self.xc[-1] == self.xb[0]:
            raise IndexError('End of compressed and beginning of buffer are same')
    #───────────────────────────────────────────────────────────────────
    def __call__(self, x_raw, y_raw, /):
        _squeezed = False # For tracking if the buffer was compressed

        if type(x_raw) != self.x_type:
            raise TypeError('Type of the x not same as the initial value')
        if type(y_raw) != self.y_type:
            raise TypeError('Type of the y not same as the initial value')

        self.xb.append(x_raw)
        self.yb.append(to_ndarray(y_raw, (-1,)))
        self.limit += 1

        G['line_buffer'].set_xdata(self.xb)
        G['line_buffer'].set_ydata(self.yb)

        if  self.limit >= self.x2: #───────────────────────────────────┐
            # Converting to numpy arrays for computations
            self.xb = to_ndarray(self.xb)
            self.yb = to_ndarray(self.yb, (self.limit + 1, -1))
            G['x'] = self.xb
            G['y'] = self.yb
            self.f2zero = self.get_f2zero(self.xb, self.yb,
                                          self.xc[-1], self.yc[-1],
                                          self.tol, self.sqrtrange,
                                          self.f_fit, self.errorfunction)
            if self.xb.shape != (self.limit + 1,):
                raise ValueError(f'xb {self.xb.shape} len {self.limit + 1}')

            if self.yb.shape != (self.limit + 1, len(self.yc[0])):
                raise ValueError(f'{self.yb.shape=}')

            self.err2, self.fit2 = self.f2zero(self.x2)

            G['xy1'], = G['ax_root'].plot(self.x1, self.err1,'g.')
            G['xy2'], = G['ax_root'].plot(self.x2, self.err2,'b.')

            if self.err2 < 0: #──────────────────────────────────────────┐

                wait('Calculating new attempt in end\n')
                G['ax_root'].plot(self.x1, self.err1,'.', color = 'black')

                self.x1, self.err1, self.fit1 = self.x2, self.err2, self.fit2
                self.x2 *= 2
                self.x2 += 1

                print(f'{self.limit=}')
                print(f'{self.x1=}\t{self.err1=}')
                print(f'{self.x2=}\t{self.err2=}')
                G['xy1'].set_xdata(self.x1)
                G['xy1'].set_ydata(self.err1)
                G['xy2'].set_xdata(self.x2)

            else: # Squeezing the buffer
                G['xy2'].set_color('red')
                wait('Points for interval found\n')
                print(f'{self._lenc=}')

                self.squeeze_buffer(self.x1, self.err1, self.x2, self.err2)
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
        return _squeezed, self._lenc, self.limit + 1
    #───────────────────────────────────────────────────────────────────
    def __len__(self):
        return self._lenc + self.limit + 1
    #───────────────────────────────────────────────────────────────────
    def __str__(self):
        return f'{self.x=} {self.y=} {self.tol=}'
    #───────────────────────────────────────────────────────────────────
    def close(self):
        self.state = 'closing'

        # Converting to numpy arrays for computations
        self.xb = to_ndarray(self.xb)
        self.yb = to_ndarray(self.yb, (self.limit + 1, -1))
        self.f2zero = f2zero.get_debug(self.xb, self.yb,
                                           self.xc[-1], self.yc[-1],
                                           self.tol, self.sqrtrange,
                                           self.f_fit, self.errorfunction)
        self.x2 = min(self.x2, self.limit)
        self.err2, self.fit2 = self.f2zero(self.x2)

        while self.err2 > 0: #───────────────────────────────────────────┐
            self.squeeze_buffer(self.x1, self.err1, self.x2, self.err2)
            self.f2zero = self.get_f2zero(self.xb, self.yb,
                                          self.xc[-1], self.yc[-1],
                                          self.tol, self.sqrtrange,
                                          self.f_fit, self.errorfunction)
            self.x2 = min(self.x2, self.limit)
            self.err2, self.fit2 = self.f2zero(self.x2)
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
    def __init__(self, x0: float, y0: float | np.ndarray,
                 tolerances: float | tuple[float, float, float] = (1e-2, 1e-3, 0),
                 initial_step: int = 100,
                 errorfunction: Callable | str = 'maxmaxabs',
                 use_numba: 0 | 1 = 0,
                 fitset: object | str = 'Poly10'):
        self.x0  = x0
        self.x_type = type(x0)
        # Variables are columns, e.G. 3xn
        self.y0  = to_ndarray(y0, (-1,))
        self.y_type = type(y0)
        self.tol = tuple([to_ndarray(tol, self.y0.shape)
                                    for tol in tolerances])

        if isinstance(errorfunction, str): #───────────────────────────┐
            self.errorfunction = errorfunctions.get(errorfunction, use_numba)
        else:
            self.errorfunction = errorfunction
        #──────────────────────────────────────────────────────────────┘
        if isinstance(fitset, str):
            self.fitset = models.get(fitset)
        else:
            self.fitset = fitset
        #──────────────────────────────────────────────────────────────┘
        self.f_fit      = self.fitset.fit[use_numba]
        self.sqrtrange  = sqrtranges[use_numba]
        self.x2         = initial_step
        self.get_f2z = f2zero.get_debug if G['debug'] else f2zero.get
    #───────────────────────────────────────────────────────────────────
    def __enter__(self):
        basic_args = (self.x0, self.y0, self.x_type, self.y_type, self.tol,
                      self.errorfunction, self.f_fit, self.sqrtrange, self.x2,
                      self.get_f2z)
        if G['debug']:
            self.record = _StreamRecord_debug(*basic_args, 
                                              self.fitset.interpolate)
        else:
            self.record = _StreamRecord(*basic_args)
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
interpolators = {'Poly10': models.get('Poly10')._interpolate}
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
        interpolator = models.get(interpolator)._interpolate
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