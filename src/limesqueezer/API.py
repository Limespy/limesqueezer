#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
API
========================================================================

Connection point for all package utilities provided
'''
import collections
import sys
import time
import types
from bisect import bisect_left as _bisect_left

import numpy as np
from matplotlib import pyplot as plt

from . import models
from . import reference as ref # Careful with this circular import
from .auxiliaries import _reset_ax
from .auxiliaries import _set_xy
from .auxiliaries import Any
from .auxiliaries import Callable
from .auxiliaries import debugsetup
from .auxiliaries import default_numba_kwargs
from .auxiliaries import Float64Array
from .auxiliaries import G
from .auxiliaries import Int64Array
from .auxiliaries import MaybeArray
from .auxiliaries import maybejit
from .auxiliaries import py_and_nb
from .auxiliaries import SqrtRange
from .auxiliaries import sqrtranges
from .auxiliaries import stats
from .auxiliaries import to_ndarray
from .auxiliaries import TolerancesInput
from .auxiliaries import TolerancesInternal
from .auxiliaries import wait
from .errorfunctions import _maxsumabs
from .errorfunctions import ErrorFunction
from .errorfunctions import errorfunctions
from .models import FitFunction
from .models import FitSet
from .models import Interpolator
from .root import _droots
from .root import _intervals
#%%═════════════════════════════════════════════════════════════════════
# @numba.jit(nopython=True,cache=True)
def n_lines(x: Float64Array,
            y: Float64Array,
            x0: float,
            y0: Float64Array,
            tol: Float64Array
            ) -> float:
    """Estimates number of lines required to fit within error tolerance."""

    if (length := len(x)) > 1:
        inds = sqrtranges[0](length - 2) # indices so that x[-1] is not included
        res = (y[-1] - y0) / (x[-1] - x0)*(x[inds] - x0).reshape([-1, 1]) - (y[inds] - y0)

        reference = _maxsumabs(res, tol)
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
        elif len_tol == 1:
            tolerances_triplet = (0., tolerances[0], 0.)
        elif len_tol == 2:
            tolerances_triplet = (*tolerances, 0.)
        else:
            tolerances_triplet = tolerances
    else:
        raise TypeError(
            'Tolerances should be a number or sequence of 1-3 numbers')
    # Relative, absolute, falloff
    return np.array([to_ndarray(tol, shape) for tol in tolerances_triplet])
#%%═════════════════════════════════════════════════════════════════════
def _tolerance(y_sample: Float64Array, tolerances: TolerancesInternal
                     ) -> Float64Array:
    y_abs = np.abs(y_sample)
    reltols = y_abs * tolerances[0]
    abstols = tolerances[1] / (tolerances[2] * y_abs + 1)
    return reltols + abstols
#───────────────────────────────────────────────────────────────────────
tolerancefunctions = py_and_nb(_tolerance)
#%%═════════════════════════════════════════════════════════════════════
# Functions to be solved in discrete root finding
F2Zero = Callable[[int], tuple[float, Float64Array]]
GetF2Zero =  Callable[[Float64Array,
                       Float64Array,
                       float,
                       Float64Array],
                      F2Zero]
#───────────────────────────────────────────────────────────────────────
def init_get_f2zero(is_debug: bool,
                    use_numba: int,
                    tol: TolerancesInternal,
                    sqrtrange: SqrtRange,
                    f_fit: FitFunction,
                    errorfunction: ErrorFunction):
    """Third orgder function to initialise the second order function to get
    f2zero."""
    tolerancefunction = tolerancefunctions[use_numba]
    if is_debug:
        def function(x: Float64Array,
                     y: Float64Array,
                     x0: float,
                     y0: Float64Array,
                     ) -> F2Zero:
            def f2zero(i: int) -> tuple[float, Float64Array]:
                """Function such that i is optimal when f2zero(i) = 0."""
                inds = sqrtrange(i)
                x_sample, y_sample = x[inds], y[inds]
                y_fit = f_fit(x_sample, y_sample, x0, y0)
                tolerance_total = tolerancefunction(y_sample, tol)

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
                _reset_ax('ax_res','Residual relative to tolerance')
                indices_x = indices_all[1:] - G['start']
                residuals_relative =  np.abs(res_all) / G['tol'] - 1
                G['ax_res'].plot(indices_x, residuals_relative,
                                    '.', color = 'blue', label = 'ignored')
                G['ax_res'].plot(inds, np.abs(residuals) / G['tol']-1,
                                    'o', color = 'red', label = 'sampled')
                G['ax_res'].legend(loc = 'lower right')
                wait('\t\tFitting\n')
                return errorfunction(y_sample, y_fit, tolerance_total), y_fit[-1]
            return f2zero
    #───────────────────────────────────────────────────────────────────
    else:
        def function(x: Float64Array,
                     y: Float64Array,
                     x0: float,
                     y0: Float64Array) -> F2Zero:
            def f2zero(i: int) -> tuple[float, Float64Array]:
                """Function such that i is optimal when f2zero(i) = 0.

                Parameters
                ----------
                i : int
                    highest index of the fit

                Returns
                -------
                tuple
                    output of the error function and last of the fit
                """
                inds = sqrtrange(i)
                x_sample, y_sample = x[inds], y[inds]
                y_fit = f_fit(x_sample, y_sample, x0, y0)
                tolerance_total = tolerancefunction(y_sample, tol)
                return errorfunction(y_sample, y_fit, tolerance_total), y_fit[-1]
            return f2zero
    return function
#%%═════════════════════════════════════════════════════════════════════
# BLOCK COMPRESSION
Compressor = Callable[[Float64Array,
                       Float64Array,
                       TolerancesInput,
                       int | None,
                       str | ErrorFunction,
                       int,
                       str | Any,
                       bool],
                      tuple[Float64Array, Float64Array]]
def LSQ10(x_in: Float64Array,
          y_in: Float64Array, /,
          tolerances: TolerancesInput = (1e-2, 1e-2, 0),
          initial_step: int  | None = None,
          errorfunction: str | ErrorFunction = 'MaxAbs',
          use_numba: int = 0,
          fitset: Any = models.Poly10,
          keepshape: bool = False
          ) -> tuple[Float64Array, Float64Array]:
    """Compresses the data of 1-dimensional system of equations i.e. single
    wait variable and one or more output variable.

    Parameters
    ----------
    x_in : Float64Array
        x-coordiantes of the points to be compressed
    y_in : Float64Array
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
    errorfunction : str, default 'MaxAbs'
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
    tuple[Float64Array, Float64Array]
        Compressed  x and Y as numpy Float64Arrays

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
    """
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
    _errorfunction = (errorfunctions[errorfunction][use_numba]
                      if isinstance(errorfunction, str)
                      else errorfunction)
    f_fit = fitset.fit[use_numba]

    # Estimation for the first offset
    if initial_step:
        offset = initial_step
    else:
        mid = end // 2
        offset = round(limit / n_lines(x[1:mid], y[1:mid], x[0], y[0], tol[1]))
    get_f2zero: GetF2Zero = init_get_f2zero(is_debug, use_numba,
                                            tol, sqrtrange,
                                            f_fit, _errorfunction)
    solver = _droots[is_debug]
    if is_debug:
        G.update(debugsetup(x, y, start_err1, fitset, start))
    #───────────────────────────────────────────────────────────────────
    # Main loop
    for _ in range(end): # Prevents infinite loop in case error
        if x[start-1] != xc[-1]:
            raise IndexError(f'Indices out of sync {start}')
        offset, fit = solver(get_f2zero(x[start:], y[start:], xc[-1], yc[-1]),
                             start_err1, offset, limit)
        step = offset + 1
        start += step # Start shifted by the number Record and the
        if start > end:
            break
        xc.append(x[start - 1])
        if is_debug: #─────────────────────────────────────────────────┐
            print(f'{start=}\t{offset=}\t{end=}\t')
            print(f'{fit=}')
            _reset_ax('ax_root', 'Maximum residual')
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
class _StreamRecord(collections.abc.Sized):
    """Class for doing stream compression for data of 1-dimensional system of
    equations i.e. single wait variable and one or more output variable."""
    def __init__(self,
                 x0: float,
                 y0: Float64Array,
                 x_type: type,
                 y_type: type,
                 tolerances: TolerancesInternal,
                 n2: int,
                 get_f2zero: GetF2Zero):
        if G['timed']: G['t_start'] = time.perf_counter()
        self.xb: list[float] = [] # Buffers for yet-to-be-recorded data
        self.yb: list[Float64Array] = []
        self.xc, self.yc = [x0], [y0]
        self.x_type: type = x_type
        self.y_type: type = y_type
        self.n1: int      = 0 # Index of starting point for looking for optimum
        self.n2: int      = n2
        self.start_err1   = -np.amax(tolerances[1]) # Default starting value
        self.state: str   = 'open' # The object is ready to accept more values
        self.limit: int   = -1 # Last index of the buffer
        self.tol          = tolerances
        self._lenc: int   = 1 # length of the Record points
        self.fit1: Float64Array = y0 # Placeholder
        self.err1         = self.start_err1 # Initialising

        self.get_f2zero = get_f2zero

        self.interval = _intervals[0]
    #─────────────────────────────────────────────────────────────────────────
    def update_f2zero(self, x_array: Float64Array, y_array: Float64Array,
                           x0: float, y0: Float64Array, limit: int):
            if x_array.shape != (limit + 1,):
                raise ValueError(f'xb {x_array.shape=} len {limit + 1}')
            if y_array.shape[0] != limit + 1:
                raise ValueError(f'yb {y_array.shape=} len {limit + 1}')

            return self.get_f2zero(x_array, y_array, x0, y0)
    #─────────────────────────────────────────────────────────────────────────
    def squeeze_buffer(self, f2zero: F2Zero, n1: int, err1: float,
                       n2: int, err2: float,
                       ) -> int:
        """Compresses the buffer by one step."""
        #──────────────────────────────────────────────────────────────┘
        offset, fit = self.interval(f2zero, n1, err1, n2, err2, self.fit1)
        self.xc.append(self.xb[offset])
        if fit is None:
            if offset == 0: # No skipping of points was possible
                self.yc.append(self.yb[offset])
            else: # Something weird happened
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
            f2zero = self.update_f2zero(np.array(self.xb),
                                        np.array(self.yb),
                                        self.xc[-1], self.yc[-1], self.limit)
            err2, fit2 = f2zero(self.n2)
            if err2 < 0: #──────────────────────────────────────────────┐
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
                        'Buffer out of sync with the compressed')
        #──────────────────────────────────────────────────────────────┘
        return was_squeezed
    #───────────────────────────────────────────────────────────────────
    def __iter__(self):
        if self.state == 'open':
            return zip(self.xc, self.yc)
        elif self.state == 'closed':
            return zip(self.x, self.y)
    #───────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.x)
    #───────────────────────────────────────────────────────────────────
    def __repr__(self):
        return f'{self.__class__.__name__}\n{self.x=}\n{self.y=}\n{self.tol=}'
    #───────────────────────────────────────────────────────────────────
    def close(self):
        """Closes the context manager."""
        self.state = 'closing'
        if self.limit != -1:
            if self.limit < 0:
                raise ValueError('Limit negative')
            # Clamping n2 to not go over the last buffer index
            if self.n2 > self.limit: self.n2 = self.limit
            # Converting to numpy arrays for fitting
            x_array = np.array(self.xb)
            y_array = to_ndarray(self.yb, (self.limit + 1, -1))
            f2zero = self.update_f2zero(x_array, y_array,
                                        self.xc[-1], self.yc[-1], self.limit)
            while (err2 := f2zero(self.n2)[0]) > 0: #──────────────────┐
                step = self.squeeze_buffer(f2zero, self.n1, self.err1,
                                            self.n2, err2)
                x_array, y_array = x_array[step:], y_array[step:]
                # Clamping n2 to not go over the last buffer index
                if self.n2 > self.limit: self.n2 = self.limit
                f2zero = self.update_f2zero(x_array, y_array,
                                            self.xc[-1], self.yc[-1], self.limit)
            #──────────────────────────────────────────────────────────┘
            self.xc.append(self.xb[-1])
            self.yc.append(self.yb[-1])
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
class _StreamRecord_debug(_StreamRecord):
    """Class for doing stream compression for data of 1-dimensional system of
    equations i.e. single wait variable and one or more output variable."""
    def __init__(self, *args, interpolator):
        if G['timed']: G['t_start'] = time.perf_counter()
        super().__init__(*args)
        self.interval = _intervals[1]
        self.max_y: float = self.yc[0][0] # For plotting
        self.min_y: float = self.yc[0][0] # For plotting
        G.update({'tol': self.tol[1],
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
    def squeeze_buffer(self, f2zero: F2Zero, n1: int, err1: float,
                       n2: int, err2: float,
                       ) -> int:
        """Compresses the buffer by one step."""

        offset, fit = self.interval(f2zero, n1, err1, n2, err2, self.fit1)
        self.xc.append(self.xb[offset])

        _reset_ax('ax_root', 'Maximum residual')

        if fit is None:
            if offset == 0: # No skipping of points was possible
                self.yc.append(self.yb[offset])
                G['x_plot'] = self.xc[-2:]
                G['y_plot'] = self.yc[-2:]
            else: # Something weird happened
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
    def __call__(self, x_raw: float, y_raw: MaybeArray) -> bool:
        was_squeezed = False # For tracking if the buffer was compressed
        if type(x_raw) != self.x_type:
            raise TypeError('Type of the x not same as the initial value')
        if type(y_raw) != self.y_type:
            raise TypeError('Type of the y not same as the initial value')
        self.xb.append(x_raw)
        self.yb.append(to_ndarray(y_raw, (-1,)))
        self.limit += 1
        _set_xy('line_buffer', self.xb, self.yb)
        G['ax_data'].set_xlim(self.xc[0], self.xb[-1]* 1.05)
        if y_raw < self.min_y: # type: ignore[operator]
            self.min_y = np.amin(y_raw)
            G['ax_data'].set_ylim(self.min_y * 1.1, self.max_y * 1.1)
        elif y_raw > self.max_y: # type: ignore[operator]
            self.max_y = np.amax(y_raw)
            G['ax_data'].set_ylim(self.min_y * 1.1, self.max_y * 1.1)

        if  self.limit >= self.n2: #───────────────────────────────────┐
            # Converting to numpy arrays for fitting
            x_array = to_ndarray(self.xb)
            y_array = to_ndarray(self.yb, (self.limit + 1, -1))
            G['x'] = x_array
            G['y'] = y_array
            f2zero, err2, fit2 = self.update_f2zero(x_array, y_array,
                                self.xc[-1], self.yc[-1], self.limit)

            G['xy1'], = G['ax_root'].plot(self.n1, self.err1,'g.')
            G['xy2'], = G['ax_root'].plot(self.n2, err2,'b.')

            if err2 < 0: #─────────────────────────────────────────────┐

                wait('Calculating new attempt in end\n')
                G['ax_root'].plot(self.n1, self.err1,'.', color = 'black')

                self.n1, self.err1, self.fit1 = self.n2, err2, fit2
                self.n2 *= 2
                self.n2 += 1

                print(f'{self.limit=}')
                print(f'{self.n1=}\t{self.err1=}')
                print(f'{self.n2=}\t{err2=}')
                _set_xy('xy1', self.n1, self.err1)
                G['xy2'].set_xdata(self.n2)

            else: # Squeezing the buffer
                G['xy2'].set_color('red')
                wait('Points for interval found\n')
                step = self.squeeze_buffer(f2zero, self.n1, self.err1,
                                           self.n2, err2)
                # Trimming the compressed section from buffer
                del self.xb[:step]
                del self.yb[:step]
                was_squeezed = True
                if self.xc[-1] == self.xb[0]:
                    raise IndexError(
                        'Buffer out of sync with the compressed')
            #──────────────────────────────────────────────────────────┘
            G['ax_data'].plot(self.xc[-1], self.yc[-1], 'go')
            wait('Next iteration\n')
        #──────────────────────────────────────────────────────────────┘
        return was_squeezed
###═════════════════════════════════════════════════════════════════════
class Stream():
    """Context manager for stream compression of data of 1 dimensional system
    of equations."""
    def __init__(self,
                 x_initial: float,
                 y_initial: float | Float64Array,
                 tolerances: TolerancesInput = (1e-2, 1e-3, 0),
                 initial_step: int = 100,
                 errorfunction: ErrorFunction | str = 'MaxAbs',
                 use_numba: int = 0,
                 fitset: FitSet = models.Poly10,
                 fragile: bool = True):
        self.x0  = x_initial
        self.x_type = type(self.x0)
        # Variables are columns, e.G. 3xn
        self.y0: Float64Array  = to_ndarray(y_initial, (-1,))
        self.y_type = type(y_initial)
        self.tol = parse_tolerances(tolerances, self.y0.shape)
        self.fragile: bool = fragile

        self.errorfunction: ErrorFunction
        if isinstance(errorfunction, str): #───────────────────────────┐
            self.errorfunction = errorfunctions[errorfunction][use_numba]
        else:
            self.errorfunction = errorfunction
        #──────────────────────────────────────────────────────────────┘
        self.fitset = fitset
        #──────────────────────────────────────────────────────────────┘
        self.f_fit      = self.fitset.fit[use_numba]
        self.sqrtrange  = sqrtranges[use_numba]
        self.n2         = initial_step
        self.get_f2zero = init_get_f2zero(G['debug'],
                                          use_numba,
                                          self.tol,
                                          self.sqrtrange,
                                          self.f_fit, self.errorfunction)
    #───────────────────────────────────────────────────────────────────
    def __enter__(self) -> _StreamRecord | _StreamRecord_debug:
        basic_args = (self.x0, self.y0, self.x_type, self.y_type, self.tol,
                      self.n2, self.get_f2zero)
        self.record: _StreamRecord | _StreamRecord_debug
        if G['debug']:
            self.record = _StreamRecord_debug(*basic_args,
                                interpolator = self.fitset.interpolate)
        else:
            self.record = _StreamRecord(*basic_args)
        return self.record
    #───────────────────────────────────────────────────────────────────
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.record.close()
        except Exception as exc:
            if self.fragile:
                raise RuntimeError('Closing of the record failed') from exc
#%%═════════════════════════════════════════════════════════════════════
# WRAPPING
# Here are the main external inteface functions
interpolators = {'Poly10': models.Poly10.interpolate}
compressors = {'LSQ10': LSQ10}
#───────────────────────────────────────────────────────────────────────
def compress(*args, compressor: Compressor | str = LSQ10, **kwargs):
    """Wrapper for easier selection of compression method."""
    if isinstance(compressor, str):
        compressor = compressors[compressor]
    return compressor(*args, **kwargs)
#───────────────────────────────────────────────────────────────────────
def decompress(x_compressed: Float64Array, y_compressed: Float64Array,
              interpolator: str | Interpolator = 'Poly10',
              use_numba: int = 0):
    """Takes array of fitting parameters and constructs whole function."""

    interpolator: Interpolator = (interpolators[interpolator][use_numba]
                                  if isinstance(interpolator, str)
                                  else interpolator)
    #───────────────────────────────────────────────────────────────────
    def _iteration(x: float, low: int = 1) -> tuple[int, MaybeArray]:
        index = _bisect_left(x_compressed, x, # type:ignore
                             lo = low, hi = y_compressed.shape[0]-1)
        return index, interpolator(x, # type:ignore
                                   *x_compressed[index-1:(index + 1)],
                                   *y_compressed[index-1:(index + 1)])
    #───────────────────────────────────────────────────────────────────
    def function(x_input):
        if hasattr(x_input, '__iter__'):
            out = np.full((len(x_input),) + y_compressed.shape[1:], np.nan)
            i_c = 1
            for i_out, x in enumerate(x_input):
                i_c, out[i_out] = _iteration(x, i_c)
            return out
        else:
            return _iteration(x_input)[1]
    #───────────────────────────────────────────────────────────────────
    return function
#%%═════════════════════════════════════════════════════════════════════
# HACKS
# A hack to make the package callable
class Pseudomodule(types.ModuleType):
    """Class that allows making the module callable."""
    @staticmethod
    def __call__(*args,
                 compressor: str | Compressor = 'LSQ10',
                 interpolator: str | Interpolator = 'Poly10',
                 **kwargs):
        """Wrapper for easier for combined compression and decompression."""
        return decompress(*compress(*args, # type:ignore
                                    compressor = compressor,
                                    **kwargs),
                          interpolator = interpolator)
#%%═════════════════════════════════════════════════════════════════════
# Here the magic happens for making the API module itself also callable
sys.modules[__name__].__class__ = Pseudomodule

__all__ = ['Stream', 'compress', 'decompress', 'Pseudomodule', 'LSQ10']
