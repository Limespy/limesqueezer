from .auxiliaries import (Callable,
                          FloatArray,
                          TolerancesInternal,
                          py_and_nb)
import numpy as np
#%%═════════════════════════════════════════════════════════════════════
## ERROR TERM
ErrorFunction = Callable[[FloatArray, FloatArray, TolerancesInternal], float]
#───────────────────────────────────────────────────────────────────────
def AbsEnd(y_sample: FloatArray,
                y_fit: FloatArray,
                tolerance_total: TolerancesInternal
                ) -> float:
    '''Maximum of absolute errors relative to tolerance

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
    return np.max(np.abs(y_fit[-1] - y_sample[-1]) - tolerance_total[-1])
#───────────────────────────────────────────────────────────────────────
def MaxAbs(y_sample: FloatArray,
                y_fit: FloatArray,
                tolerance_total: TolerancesInternal
                ) -> float:
    '''Maximum of absolute errors relative to tolerance

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
    return np.max(np.abs(y_fit - y_sample) - tolerance_total)
#───────────────────────────────────────────────────────────────────────
def MaxMAbs(y_sample: FloatArray,
                y_fit: FloatArray,
                tolerance_total: TolerancesInternal
                ) -> float:
    '''Maximum of mean excess errors relative to tolerance or maximum of the end values.

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
    residuals = np.abs(y_fit - y_sample)
    return np.amax(np.mean(residuals - tolerance_total, 0))
#───────────────────────────────────────────────────────────────────────
def MaxMAbs_AbsEnd(y_sample: FloatArray,
                y_fit: FloatArray,
                tolerance_total: TolerancesInternal
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
    residuals = np.abs(y_fit - y_sample)
    excess = residuals - tolerance_total
    excess_end = np.amax(excess[-1:])
    excess_mean = np.amax(np.mean(excess, 0))
    return excess_end if excess_end > excess_mean else excess_mean
#───────────────────────────────────────────────────────────────────────
def MaxMS(y_sample: FloatArray,
                y_fit: FloatArray,
                tolerance_total: TolerancesInternal
                ) -> float:
    '''Root mean square error.
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
    return np.amax(np.mean(residuals * residuals - tolerance_total, 0))
#───────────────────────────────────────────────────────────────────────
def MaxMS_SEnd(y_sample: FloatArray,
            y_fit: FloatArray,
            tolerance_total: TolerancesInternal
            ) -> float:
    '''Intended to clamp the end point within absolute value of tolerance for more stability. Returns bigger of:
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

    residuals = y_fit - y_sample
    # Excess at the last point
    residuals *= residuals
    excess = residuals - tolerance_total
    excess_end = np.amax(excess[-1:])
    excess_mean = np.amax(np.mean(excess, 0))
    return excess_end if excess_end > excess_mean else excess_mean
#%%═════════════════════════════════════════════════════════════════════
def maxsumabs(residuals: FloatArray, tolerance: float | FloatArray) -> float:
    return np.amax(np.sum(np.abs(residuals) - tolerance) / tolerance)
#───────────────────────────────────────────────────────────────────────

errorfunctions = {'AbsEnd': py_and_nb(MaxAbs),
                  'MaxAbs': py_and_nb(MaxAbs),
                  'MaxMAbs': py_and_nb(MaxMAbs),
                  'MaxMAbs_AbsEnd': py_and_nb(MaxMAbs_AbsEnd),
                  'MaxMS': py_and_nb(MaxMS),
                  'MaxMS_SEnd': py_and_nb(MaxMS_SEnd)
                  }
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