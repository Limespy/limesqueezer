#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
API
========================================================================

Connection point for all package utilities provided
'''
from .auxiliaries import (to_ndarray,
                          wait,
                          sqrtranges,
                          SqrtRange,
                          debugsetup,
                          stats)
from . import errorfunctions
from .errorfunctions import ErrorFunction
from .GLOBALS import (G,
                      FloatArray,
                      MaybeArray,
                      Any,
                      Callable,
                      TolerancesInput,
                      TolerancesInternal) # Type signatures
from . import models
from .models import FitFunction, Interpolator # Type signatures
from . import reference as ref # Careful with this circular import
from .root import droot, droot_debug, interval, interval_debug

from bisect import bisect_left
import collections
from matplotlib import pyplot as plt
import numpy as np

import sys
import time
import types
#%%═════════════════════════════════════════════════════════════════════
# @numba.jit(nopython=True,cache=True)
def n_lines(x: FloatArray,
            y: FloatArray,
            x0: float,
            y0: FloatArray,
            tol: FloatArray
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
def parse_tolerances(tolerances: TolerancesInput, shape: tuple[int, ...]
                     ) -> TolerancesInternal:
    if isinstance(tolerances, (float, np.ndarray)):
        tolerances_triplet: tuple[MaybeArray, ...]  = (0., tolerances, 0.)
    elif isinstance(tolerances, (tuple, list)):
        len_tol = len(tolerances)
        if not (0 < len_tol < 4):
            raise ValueError(f'Tolerances length should be 1-3, was {len_tol}')
        elif len(tolerances) == 1:
            tolerances_triplet = (0., tolerances[0], 0.)
        elif len(tolerances) == 2:
            tolerances_triplet = (*tolerances, 0.)
        else:
            tolerances_triplet = tolerances
    else:
        raise TypeError(
            'Tolerances should be a number or sequence of 1-3 numbers')
    # Relative, absolute, falloff
    return (to_ndarray(tolerances_triplet[0], shape),
            to_ndarray(tolerances_triplet[1], shape),
            to_ndarray(tolerances_triplet[2], shape))
#%%═════════════════════════════════════════════════════════════════════
# Functions to be solved in discrete root finding
F2Zero = Callable[[int], tuple[float, FloatArray]]
GetF2Zero =  Callable[[FloatArray,
                       FloatArray,
                       float,
                       FloatArray,
                       TolerancesInternal,
                       SqrtRange,
                       FitFunction,
                       ErrorFunction],
                      F2Zero]
#───────────────────────────────────────────────────────────────────────
def get_f2zero(x: FloatArray,
        y: FloatArray,
        x0: float,
        y0: FloatArray,
        tol: TolerancesInternal,
        sqrtrange: SqrtRange,
        f_fit: FitFunction,
        errorfunction: ErrorFunction
        ) -> F2Zero:
    def f2zero(i: int) -> tuple[float, FloatArray]:
        '''Function such that i is optimal when f2zero(i) = 0

        Parameters
        ----------
        i : int
            highest index of the fit 

        Returns
        -------
        tuple
            output of the error function and last of the fit
        '''
        inds = sqrtrange(i)
        x_sample, y_sample = x[inds], y[inds]
        y_fit = f_fit(x_sample, y_sample, x0, y0)
        return errorfunction(y_sample, y_fit, tol), y_fit[-1]
    return f2zero
#───────────────────────────────────────────────────────────────────────
def get_f2zero_debug(x: FloatArray,
              y: FloatArray,
              x0: float,
              y0: FloatArray,
              tol: TolerancesInternal,
              sqrtrange: Callable[[int], np.int64],
              f_fit: FitFunction,
              errorfunction: ErrorFunction
              ) -> Callable[[int], tuple[float, FloatArray]]:
    def f2zero(i: int) -> tuple[float, FloatArray]:
        '''Function such that i is optimal when f2zero(i) = 0'''
        inds = sqrtrange(i)
        x_sample, y_sample = x[inds], y[inds]
        y_fit = f_fit(x_sample, y_sample, x0, y0)
        residuals = y_fit - y_sample
        if len(residuals) == 1:
            print(f'\t\t{residuals=}')
        print(f'\t\tstart = {G["start"]} end = {i + G["start"]} points = {i + 1}')
        print(f'\t\tx0\t{x0}\n\t\tx[0]\t{x[inds][0]}\n\t\tx[-1]\t{x[inds][-1]}\n\t\txstart = {G["x"][G["start"]]}')
        indices_all = np.arange(-1, i) + G['start']
        G['x_plot'] = G['x'][indices_all]
        G['y_plot'] = G['interp'](G['x_plot'], x0, x[inds][-1], y0, y_fit[-1])
        # print(f'{G["y_plot"].shape=}')
        G['line_fit'].set_xdata(G['x_plot'])
        G['line_fit'].set_ydata(G['y_plot'])
        # print(f'{G["y"][indices_all].shape=}')
        res_all = G['y_plot'][1:] - G['y'][indices_all].flatten()[1:]
        print(f'\t\t{residuals.shape=}\n\t\t{res_all.shape=}')
        G['ax_res'].clear()
        G['ax_res'].grid()
        G['ax_res'].axhline(color = 'red', linestyle = '--')
        G['ax_res'].set_ylabel('Residual relative to tolerance')
        indices_x = indices_all[1:] - G['start']
        residuals_relative =  np.abs(res_all) / G['tol'] - 1
        G['ax_res'].plot(indices_x, residuals_relative,
                            '.', color = 'blue', label = 'ignored')
        G['ax_res'].plot(inds, np.abs(residuals) / G['tol']-1,
                            'o', color = 'red', label = 'sampled')
        G['ax_res'].legend(loc = 'lower right')
        wait('\t\tFitting\n')
        return errorfunction(y_sample, y_fit, tol), y_fit[-1]
    return f2zero
#%%═════════════════════════════════════════════════════════════════════
# BLOCK COMPRESSION
Compressor = Callable[[FloatArray,
                       FloatArray,
                       TolerancesInput,
                       int | None,
                       str | ErrorFunction,
                       int,
                       str | Any,
                       bool],
                      tuple[FloatArray, FloatArray]]
def LSQ10(x_in: FloatArray,
          y_in: FloatArray, /,
          tolerances: TolerancesInput = (1e-2, 1e-2, 0),
          initial_step: int | None  = None,
          errorfunction: str | ErrorFunction = 'maxmaxabs',
          use_numba: int = 0,
          fitset: str | Any   = 'Poly10',
          keepshape: bool = False
          ) -> tuple[FloatArray, FloatArray]:
    '''Compresses the data of 1-dimensional system of equations
    i.e. single wait variable and one or more output variable

    Parameters
    ----------
    x_in : FloatArray
        x-coordiantes of the points to be compressed
    y_in : FloatArray
        y-coordinates of the points to be compressed
    tolerances : tuple, default (1e-2, 1e-2, 0)
        tolerances, Falloff determines how much the absolute error is
        reuduced as y value grows.
            If 3 values: (relative, absolute, falloff)
            If 1 values: (relative, absolute, 0)
            If 1 value:  (0, absolute, 0)
    initial_step : int, default None
        First compression step to be calculated.
        If None, it is automatically calculated.
        Provide for testing purposes or for miniscule reduction in setup time
    errorfunction : str, default 'maxmaxabs'
        Function which is used to compute the error of the fit, by 
    use_numba : int, default 0
        For using functions with Numba JIT compilation set as 1
    fitset : str, default 'Poly10'
        Name of the fitting function set
    keepshape : bool, default False
        Whether the output is in similar shape to input
        or the compressor gets to choose

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Compressed  x and Y as numpy FloatArrays

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
    tol = parse_tolerances(tolerances, y[0].shape)

    xc, yc = [x[0]], [y[0]]
    # Starting value for discrete root calculation
    start_err1 = - np.amax(tol[1]) 

    start   = 1 # Index of starting point for looking for optimum
    end     = xlen - 1 # Number of datapoints -1, i.e. the last index
    limit   = end - start

    sqrtrange = sqrtranges[use_numba]
    _errorfunction = (errorfunctions.get(errorfunction, use_numba)
                      if isinstance(errorfunction, str)
                      else errorfunction)
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
        G.update(debugsetup(x, y, start_err1, fitset, start))
        get_f2z = get_f2zero_debug
        solver = droot_debug
    else:
        get_f2z = get_f2zero
        solver = droot
    #───────────────────────────────────────────────────────────────────
    # Main loop
    for _ in range(end): # Prevents infinite loop in case error
        if x[start-1] != xc[-1]:
            raise IndexError(f'Indices out of sync {start}')
        offset, fit = solver(get_f2z(x[start:], y[start:], xc[-1], yc[-1],
                                     tol, sqrtrange, f_fit, _errorfunction),
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
                raise RuntimeError(
                    'Fit returned was None, check fit functions for errors')
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
def init_update_f2zero(f2zero_init: GetF2Zero,
                       tolerances: TolerancesInternal,
                       sqrtrange: SqrtRange,
                       fit: FitFunction,
                       errorfunction: ErrorFunction):
    def update(x_array: FloatArray, y_array: FloatArray,
               x0: float, y0: FloatArray, n: int, limit):
        if x_array.shape != (limit + 1,):
            raise ValueError(f'xb {x_array.shape} len {limit + 1}')
        if y_array.shape != (limit + 1, 1):
            raise ValueError(f'{y_array.shape=}')

        f2zero = f2zero_init(x_array, y_array, x0, y0,
                             tolerances, sqrtrange, fit, errorfunction)
        return f2zero, *f2zero(n)
    return update
 #───────────────────────────────────────────────────────────────────
class _StreamRecord(collections.abc.Sized):
    """Class for doing stream compression for data of 1-dimensional
    system of equations
    i.e. single wait variable and one or more output variable
    """
    def __init__(self,
                 x0: float,
                 y0: FloatArray,
                 x_type: type,
                 y_type: type,
                 tolerances: TolerancesInternal,
                 errorfunction: ErrorFunction,
                 f_fit: FitFunction,
                 sqrtrange: SqrtRange,
                 n2: int,
                 get_f2zero: GetF2Zero):
        if G['timed']: G['t_start'] = time.perf_counter()
        self.xb: list[float] = [] # Buffers for yet-to-be-recorded data
        self.yb: list[FloatArray] = [] 
        self.xc, self.yc = [x0], [y0]
        self.x_type, self.y_type = x_type, y_type
        self.n1: int = 0 # Index of starting point for looking for optimum
        self.n2         = n2
        self.tol        = tolerances
        self.start_err1   = -np.amax(tolerances[1]) # Default starting value
        self.state      = 'open' # The object is ready to accept more values
        self.limit: int      = -1 # Last index of the buffer
        self._lenc: int      = 1 # length of the Record points
        self.fit1: FloatArray = y0 # Placeholder
        self.err1         = self.start_err1 # Initialising
        self._update_f2zero = init_update_f2zero(get_f2zero, tolerances,
                                                 sqrtrange, f_fit,
                                                 errorfunction)
    # #───────────────────────────────────────────────────────────────────
    def squeeze_buffer(self, f2zero: F2Zero, x1: float, err1: float,
                       n2: float, err2: float,
                       ):
        '''Compresses the buffer by one step'''
        #──────────────────────────────────────────────────────────────┘
        offset, fit = interval(f2zero, x1, err1, n2, err2, self.fit1)
        self.xc.append(self.xb[offset])
        if fit is None:
            if offset == 0: # No skipping of points was possible
                self.yc.append(self.yb[offset])
            else: # Something weird
                raise RuntimeError('Fit not found')
        else:
            self.yc.append(fit)
        self.n1, self.err1, step = 0, self.start_err1, offset + 1

        self.limit -= step
        self._lenc += 1

        self.n2 = offset # Approximation
        return step
    #───────────────────────────────────────────────────────────────────
    def __call__(self, x_raw: float, y_raw: MaybeArray) -> bool:
        was_squeezed = False # For tracking if the buffer was compressed

        if type(x_raw) != self.x_type:
            raise TypeError('Type of the x not same as the initial value')
        if type(y_raw) != self.y_type:
            raise TypeError('Type of the y not same as the initial value')

        self.xb.append(x_raw)
        self.yb.append(to_ndarray(y_raw, (-1,)))
        self.limit += 1
        if  self.limit >= self.n2: #───────────────────────────────────┐
            # Converting to numpy arrays for fitting
            f2zero, err2, fit2 = self._update_f2zero(to_ndarray(self.xb),
                                to_ndarray(self.yb, (self.limit + 1, -1)),
                                self.xc[-1], self.yc[-1], self.n2, self.limit)

            if err2 < 0: #──────────────────────────────────────────┐
                self.n1, self.err1, self.fit1 = self.n2, err2, fit2
                self.n2 *= 2
                self.n2 += 1
            else: # Squeezing the buffer
                step = self.squeeze_buffer(f2zero, self.n1, self.err1,
                                           self.n2, err2)
                # Trimming the compressed section from buffer
                del self.xb[:step]
                del self.yb[:step]
                was_squeezed = True

            if self.xc[-1] == self.xb[0]:
                raise IndexError(
                    'End of compressed and beginning of buffer are same')
        #──────────────────────────────────────────────────────────────┘
        return was_squeezed
    #───────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.x)
    #───────────────────────────────────────────────────────────────────
    def __str__(self):
        return f'{self.x=} {self.y=} {self.tol=}'
    #───────────────────────────────────────────────────────────────────
    def close(self):
        '''Closes the context manager'''
        self.state = 'closing'
        if self.limit != -1:
            # Clamping n2 to not go over the last buffer index
            if self.n2 > self.limit: self.n2 = self.limit
            # Converting to numpy arrays for fitting
            x_array = to_ndarray(self.xb)
            if self.limit < 0:
                raise ValueError
            y_array = to_ndarray(self.yb, (self.limit + 1, -1))
            f2zero, err2, _ = self._update_f2zero(x_array, y_array,
                                                  self.xc[-1], self.yc[-1],
                                                  self.n2, self.limit)
            while err2 > 0: #──────────────────────────────────────────┐
                step = self.squeeze_buffer(f2zero, self.n1, self.err1,
                                           self.n2, err2)
                x_array, y_array = x_array[step:], y_array[step:]
                # Clamping n2 to not go over the last buffer index
                if self.n2 > self.limit: self.n2 = self.limit
                f2zero, err2, _ = self._update_f2zero(x_array, y_array,
                                                      self.xc[-1], self.yc[-1],
                                                      self.n2, self.limit)
            #──────────────────────────────────────────────────────────┘
            self.xc.append(self.xb[-1])
            self.yc.append(to_ndarray(self.yb[-1], (1,)))
            self._lenc += 1
        # Final packing and cleaning
        self.x = to_ndarray(self.xc, (self._lenc,))
        self.y = to_ndarray(self.yc, (self._lenc , -1))
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
                 y0: FloatArray,
                 x_type: type,
                 y_type: type,
                 tolerances: TolerancesInternal,
                 errorfunction: ErrorFunction,
                 f_fit: FitFunction,
                 sqrtrange: SqrtRange,
                 n2: int,
                 get_f2zero: GetF2Zero,
                 interpolator):
        if G['timed']: G['t_start'] = time.perf_counter()
        self.xb: list[float] = [] # Buffers for yet-to-be-recorded data
        self.yb: list[FloatArray] = [] 
        self.xc, self.yc = [x0], [y0]
        self.x_type, self.y_type = x_type, y_type
        self.n1: int    = 0 # Index of starting point for looking for optimum
        self.n2: int         = n2
        self.start_err1   = -np.amax(tolerances[1]) # Default starting value
        self.state      = 'open' # The object is ready to accept more values
        self.limit: int      = -1 # Last index of the buffer
        self.tol = tolerances
        self._lenc: int      = 1 # length of the Record points
        self.fit1: FloatArray = y0 # Placeholder
        self.err1         = self.start_err1 # Initialising
        self.get_f2zero = get_f2zero
        self._update_f2zero = init_update_f2zero(get_f2zero, tolerances,
                                                 sqrtrange, f_fit,
                                                 errorfunction)
        self.max_y: float = y0[0] # For plotting
        self.min_y: float = y0[0] # For plotting
        G.update({'tol': tolerances[1],
                    'x': np.array(self.xb),
                    'y': np.array(self.yb),
                    'xc': self.xb,
                    'yc': self.yb,
                    'interp': interpolator,
                    'limit': self.limit,
                    'start': 1})
        G['fig'], (G['ax_data'], G['ax_res'], G['ax_root']) = plt.subplots(3,1)
        G['line_buffer'], = G['ax_data'].plot(0, 0, 'b-',
                                                label = 'buffer')
        G['line_fit'], = G['ax_data'].plot(0, 0, '-', color = 'orange',
                                                label = 'fit')

        G['ax_root'].set_ylabel('Tolerance left')

        plt.ion()
        plt.show()
        wait('Initialised')
    #───────────────────────────────────────────────────────────────────
    def squeeze_buffer(self, f2zero, x1, err1, n2, err2):
        '''Compresses the buffer by one step'''

        offset, fit = interval_debug(f2zero, x1, err1, n2, err2, self.fit1)
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
            G['x_plot'] = self.xb[:offset + 1]
            G['y_plot'] = G['interp'](G['x_plot'], *self.xc[-2:], *self.yc[-2:])

        G['ax_data'].plot(G['x_plot'], G['y_plot'], color = 'red')

        self.n1, self.err1, step = 0, self.start_err1, offset + 1

        self.limit -= step
        self._lenc += 1

        self.n2 = offset # Approximation
        return step
    #───────────────────────────────────────────────────────────────────
    def __call__(self, x_raw: float, y_raw: float) -> bool:
        was_squeezed = False # For tracking if the buffer was compressed
        if type(x_raw) != self.x_type:
            raise TypeError('Type of the x not same as the initial value')
        if type(y_raw) != self.y_type:
            raise TypeError('Type of the y not same as the initial value')
        self.xb.append(x_raw)
        self.yb.append(to_ndarray(y_raw, (-1,)))
        self.limit += 1
        G['line_buffer'].set_xdata(self.xb)
        G['line_buffer'].set_ydata(self.yb)
        G['ax_data'].set_xlim(self.xc[0], self.xb[-1]* 1.05)
        if y_raw < self.min_y:
            self.min_y = y_raw
            G['ax_data'].set_ylim(self.min_y * 1.1, self.max_y * 1.1)
        elif y_raw > self.max_y:
            self.max_y = y_raw
            G['ax_data'].set_ylim(self.min_y * 1.1, self.max_y * 1.1)

        if  self.limit >= self.n2: #───────────────────────────────────┐
            # Converting to numpy arrays for fitting
            x_array = to_ndarray(self.xb)
            y_array = to_ndarray(self.yb, (self.limit + 1, -1))
            G['x'] = x_array
            G['y'] = y_array
            f2zero, err2, fit2 = self._update_f2zero(x_array, y_array,
                                self.xc[-1], self.yc[-1], self.n2, self.limit)

            G['xy1'], = G['ax_root'].plot(self.n1, self.err1,'g.')
            G['xy2'], = G['ax_root'].plot(self.n2, err2,'b.')

            if err2 < 0: #──────────────────────────────────────────┐

                wait('Calculating new attempt in end\n')
                G['ax_root'].plot(self.n1, self.err1,'.', color = 'black')

                self.n1, self.err1, self.fit1 = self.n2, err2, fit2
                self.n2 *= 2
                self.n2 += 1

                print(f'{self.limit=}')
                print(f'{self.n1=}\t{self.err1=}')
                print(f'{self.n2=}\t{err2=}')
                G['xy1'].set_xdata(self.n1)
                G['xy1'].set_ydata(self.err1)
                G['xy2'].set_xdata(self.n2)

            else: # Squeezing the buffer
                G['xy2'].set_color('red')
                wait('Points for interval found\n')
                step = self.squeeze_buffer(f2zero, self.n1, self.err1,
                                           self.n2, err2)
                # Trimming the compressed section from buffer
                del self.xb[:step]
                del self.yb[:step]
                if self.xc[-1] == self.xb[0]:
                    raise IndexError(
                        'End of compressed and beginning of buffer are same')
            #──────────────────────────────────────────────────────────┘
            G['ax_data'].plot(self.xc[-1], self.yc[-1], 'go')
            wait('Next iteration\n')
        #──────────────────────────────────────────────────────────────┘
        return was_squeezed
    #───────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.x)
    #───────────────────────────────────────────────────────────────────
    def __str__(self):
        return f'{self.x=} {self.y=} {self.tol=}'
    #───────────────────────────────────────────────────────────────────
    def close(self):
        '''Closes the context manager'''
        self.state = 'closing'
        if self.limit != -1:
            # Clamping n2 to not go over the last buffer index
            if self.n2 > self.limit: self.n2 = self.limit
            # Converting to numpy arrays for fitting
            x_array = to_ndarray(self.xb)
            y_array = to_ndarray(self.yb, (self.limit + 1, -1))
            f2zero, err2, _ = self._update_f2zero(x_array, y_array,
                                self.xc[-1], self.yc[-1], self.n2, self.limit)
            while err2 > 0: #─────────────────────────────────────────┐
                step = self.squeeze_buffer(f2zero, self.n1, self.err1, self.n2, err2)
                x_array, y_array = x_array[step:], y_array[step:]
                # Clamping n2 to not go over the last buffer index
                if self.n2 > self.limit: self.n2 = self.limit
                f2zero, err2, _ = self._update_f2zero(x_array, y_array,
                                self.xc[-1], self.yc[-1], self.n2, self.limit)
            #──────────────────────────────────────────────────────────────┘
            self.xc.append(self.xb[-1])
            self.yc.append(to_ndarray(self.yb[-1], (1,)))
            self._lenc += 1
        # Final packing and cleaning
        self.x = to_ndarray(self.xc, (self._lenc,))
        self.y = to_ndarray(self.yc, (self._lenc , -1))
        plt.ioff()
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
    def __init__(self,
                 x_initial: float,
                 y_initial: float | FloatArray,
                 tolerances: TolerancesInput = (1e-2, 1e-3, 0),
                 initial_step: int = 100,
                 errorfunction: ErrorFunction | str = 'maxmaxabs',
                 use_numba: int = 0,
                 fitset: object | str = 'Poly10'):
        self.x0  = x_initial
        self.x_type = type(self.x0)
        # Variables are columns, e.G. 3xn
        self.y0  = to_ndarray(y_initial, (-1,))
        self.y_type = type(y_initial)
        self.tol = parse_tolerances(tolerances, self.y0.shape)

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
        self.n2         = initial_step
        self.get_f2z = get_f2zero_debug if G['debug'] else get_f2zero
    #───────────────────────────────────────────────────────────────────
    def __enter__(self):
        basic_args = (self.x0, self.y0, self.x_type, self.y_type, self.tol,
                      self.errorfunction, self.f_fit, self.sqrtrange, self.n2,
                      self.get_f2z)
        if G['debug']:
            self.record: _StreamRecord_debug | _StreamRecord = _StreamRecord_debug(*basic_args,
                                              self.fitset._interpolate)
        else:
            self.record = _StreamRecord(*basic_args)
        return self.record
    #───────────────────────────────────────────────────────────────────
    def __exit__(self, exc_type, exc_value, traceback):
        self.record.close()

#%%═════════════════════════════════════════════════════════════════════
def _decompress(x_compressed: FloatArray,
               fit_array: FloatArray,
               interpolator: Interpolator):
    '''Takes array of fitting parameters and constructs whole function'''
    #───────────────────────────────────────────────────────────────────
    def _iteration(x: float, low: int = 1):
        index = bisect_left(x_compressed, x,
                            lo = low, hi = fit_array.shape[0]-1) # type:ignore
        return index, interpolator(x, *x_compressed[index-1:(index + 1)],
                                   *fit_array[index-1:(index + 1)])
    #───────────────────────────────────────────────────────────────────
    def function(x_input):
        if hasattr(x_input, '__iter__'):
            out = np.full((len(x_input),) + fit_array.shape[1:], np.nan)
            i_c = 1
            for i_out, x in enumerate(x_input):
                i_c, out[i_out] = _iteration(x, i_c)
            return out
        else:
            return _iteration(x_input)[1]
    #───────────────────────────────────────────────────────────────────
    return function
#%%═════════════════════════════════════════════════════════════════════
# WRAPPING
# Here are the main external inteface functions
compressors = {'LSQ10': LSQ10}
interpolators = {'Poly10': models.get('Poly10')._interpolate}
#───────────────────────────────────────────────────────────────────────
def compress(*args, compressor: str | Compressor = 'LSQ10', **kwargs):
    '''Wrapper for easier selection of compression method'''
    if isinstance(compressor, str):
        try:
            compressor = compressors[compressor]
        except KeyError:
            raise NotImplementedError(f'{compressor} not in the dictionary of builtin compressors')
    return compressor(*args, **kwargs)
#───────────────────────────────────────────────────────────────────────
def decompress(x: FloatArray, y: FloatArray,
              interpolator: str | Interpolator = 'Poly10'):
    '''Wrapper for easier selection of compression method'''
    if isinstance(interpolator, str):
        return _decompress(x, y, models.get(interpolator)._interpolate)
    return _decompress(x, y, interpolator)
#%%═════════════════════════════════════════════════════════════════════
# HACKS
# A hack to make the package callable
class Pseudomodule(types.ModuleType):
    """Class that wraps the individual plotting functions
    an allows making the module callable"""
    @staticmethod
    def __call__(*args,
                 compressor: str | Compressor = 'LSQ10',
                 interpolator: str | Interpolator = 'Poly10',
                 **kwargs):
        '''Wrapper for easier for combined compression and decompression'''
        return decompress(*compress(*args, 
                                    compressor = compressor,
                                    **kwargs), # type:ignore
                          interpolator = interpolator)
#%%═════════════════════════════════════════════════════════════════════
# Here the magic happens for making the API module itself also callable
sys.modules[__name__].__class__ = Pseudomodule