from .auxiliaries import Callable, FloatArray, TolerancesInternal, default_numba_kwargs, maybejit, py_and_nb
import numpy as np
import numba as nb
#%%═════════════════════════════════════════════════════════════════════
def _tolerance(y_sample: FloatArray, tolerances: TolerancesInternal
                     ) -> FloatArray:
    y_abs = np.abs(y_sample)
    reltols = y_abs * tolerances[0]
    abstols = tolerances[1] / (tolerances[2] * y_abs + 1)
    return reltols + abstols
#───────────────────────────────────────────────────────────────────────
_tolerancefunctions = py_and_nb(_tolerance)
#%%═════════════════════════════════════════════════════════════════════
## ERROR TERM
ErrorFunction = Callable[[FloatArray, FloatArray, TolerancesInternal], float]
function_names = ('maxmaxabs', 'maxMS', 'maxMS_SqrEnd')
#───────────────────────────────────────────────────────────────────────
def _setup(function_name: str, use_numba: int) -> ErrorFunction:
    '''Second order function to set up both python and numba versions
    of error functions

    Parameters
    ----------
    function_name : str
        name of the function to be set up
    use_numba : int
        whether generate numba version or not

    Returns
    -------
    ErrorFunction
        _description_

    Raises
    ------
    KeyError
        Function name does not match predetermined set
    '''
    tolerance = _tolerancefunctions[use_numba]
    #───────────────────────────────────────────────────────────────────
    if function_name == function_names[0]:
        def function(y_sample: FloatArray,
                     y_fit: FloatArray,
                     tolerances: TolerancesInternal
                     ) -> float:
            '''Maximum of absolute orrors relative to tolerance

            Parameters
            ----------
            y_sample : FloatArray
                Y values of points of data selected for error calculation
            y_fit : FloatArray
                Y values from fitting the model into data
            tolerances : TolerancesInternal
                Tolerances for errors
                1) Relative error array
                2) Absolute error array
                3) Falloff array

            Returns
            -------
            float
                Error value. Should be <0 for fit to be acceptable
            '''
            return np.max(np.abs(y_fit - y_sample) - tolerance(y_sample, tolerances))
    #───────────────────────────────────────────────────────────────────
    elif function_name == function_names[1]:
        def function(y_sample: FloatArray,
                     y_fit: FloatArray,
                     tolerances: TolerancesInternal
                     ) -> float:
            '''Root mean square error without numba.
            1. Calculate residuals squared
            2. Square root of mean along a column
            3. Find largest of those difference to tolerance

            Parameters
            ----------
            y_sample : FloatArray
                Y values of points of data selected for error calculation
            y_fit : FloatArray
                Y values from fitting the model into data
            tolerances : TolerancesInternal
                Tolerances for errors
                1) Relative tolerance array
                2) Absolute tolerance array
                3) Falloff array

            Returns
            -------
            float
                Error value. Should be <0 for fit to be acceptable
            '''
            residuals = y_fit - y_sample
            return np.amax(np.mean(residuals * residuals - tolerance(y_sample, tolerances), 0))
    #───────────────────────────────────────────────────────────────────
    elif function_name == function_names[2]:
        def function(y_sample: FloatArray,
                     y_fit: FloatArray,
                     tolerances: TolerancesInternal
                     ) -> float:
            '''Without Numba. Intended to clamp the end point within absolute value of tolerance for more stability. Returns bigger of:
            - root mean square error
            - end point maximum absolute error
            1. Calculate endpoint maximum absolute error
            2. Calculate residuals squared

            Parameters
            ----------
            y_sample : FloatArray
                Y values of points of data selected for error calculation
            y_fit : FloatArray
                Y values from fitting the model into data
            tolerances : TolerancesInternal
                Tolerances for errors
                1) Relative error array
                2) Absolute error array
                3) Falloff array

            Returns
            -------
            float
                Error value. Should be <0 for fit to be acceptable
            '''

            tols = tolerance(y_sample, tolerances)

            residuals = y_fit - y_sample
            # Excess at the last point
            residuals *= residuals
            excess = residuals - tols
            excess_end = np.max(excess[-1:])
            excess_max_MS = np.amax(np.mean(excess, 0))
            return excess_end if excess_end > excess_max_MS else excess_max_MS
    #───────────────────────────────────────────────────────────────────
    else:
        raise KeyError(f'Error function {function_name} not reconised')
    function = maybejit(use_numba, function, **default_numba_kwargs)
    function.__name__ = function_name
    return function
#%%═════════════════════════════════════════════════════════════════════
def maxsumabs(residuals: FloatArray, tolerance: float | FloatArray) -> float:
    return np.amax(np.sum(np.abs(residuals) - tolerance) / tolerance)
#───────────────────────────────────────────────────────────────────────
errorfunctions = {name: (_setup(name, 0), _setup(name, 1))
                  for name in function_names}
#───────────────────────────────────────────────────────────────────────
def get(name, use_numba):
    try:
        versions = errorfunctions[name]
    except KeyError:
        raise NotImplementedError(f'Error function {name} not recognised. Valid are {tuple(errorfunctions.keys())}')
    try:
        return versions[use_numba]
    except IndexError:
        raise ValueError(f'Numba selector should be 0 or 1, not {use_numba}')