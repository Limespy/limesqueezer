import collections

import numpy as np

from .auxiliaries import Callable
from .auxiliaries import Float64Array
from .auxiliaries import py_and_nb
from .auxiliaries import TolerancesInternal
#%%═════════════════════════════════════════════════════════════════════
## ERROR TERM
ErrorFunction = Callable[[Float64Array, Float64Array, TolerancesInternal], float]
#───────────────────────────────────────────────────────────────────────
def AbsEnd(y_sample: Float64Array,
                y_fit: Float64Array,
                tolerance_total: TolerancesInternal
                ) -> float:
    """Maximum of absolute errors relative to tolerance.

    Parameters
    ----------
    y_sample : Float64Array
        Y values of points of data selected for error calculation
    y_fit : Float64Array
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
    """
    return np.max(np.abs(y_fit[-1] - y_sample[-1]) - tolerance_total[-1])
#───────────────────────────────────────────────────────────────────────
def MaxAbs(y_sample: Float64Array,
                y_fit: Float64Array,
                tolerance_total: TolerancesInternal
                ) -> float:
    """Maximum of absolute errors relative to tolerance.

    Parameters
    ----------
    y_sample : Float64Array
        Y values of points of data selected for error calculation
    y_fit : Float64Array
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
    """
    return np.max(np.abs(y_fit - y_sample) - tolerance_total)
#───────────────────────────────────────────────────────────────────────
def MaxMAbs(y_sample: Float64Array,
                y_fit: Float64Array,
                tolerance_total: TolerancesInternal
                ) -> float:
    """Maximum of mean excess errors relative to tolerance or maximum of the
    end values.

    Parameters
    ----------
    y_sample : Float64Array
        Y values of points of data selected for error calculation
    y_fit : Float64Array
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
    """
    residuals = np.abs(y_fit - y_sample)
    return np.amax(np.mean(residuals - tolerance_total, 0))
#───────────────────────────────────────────────────────────────────────
def MaxMAbs_AbsEnd(y_sample: Float64Array,
                y_fit: Float64Array,
                tolerance_total: TolerancesInternal
                ) -> float:
    """Maximum of absolute orrors relative to tolerance.

    Parameters
    ----------
    y_sample : Float64Array
        Y values of points of data selected for error calculation
    y_fit : Float64Array
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
    """
    residuals = np.abs(y_fit - y_sample)
    excess = residuals - tolerance_total
    excess_end = np.amax(excess[-1:])
    excess_mean = np.amax(np.mean(excess, 0))
    return excess_end if excess_end > excess_mean else excess_mean
#───────────────────────────────────────────────────────────────────────
def MaxMS(y_sample: Float64Array,
                y_fit: Float64Array,
                tolerance_total: TolerancesInternal
                ) -> float:
    '''Root mean square error.
    1. Calculate residuals squared
    2. Square root of mean along a column
    3. Find largest of those difference to tolerance

    Parameters
    ----------
    y_sample : Float64Array
        Y values of points of data selected for error calculation
    y_fit : Float64Array
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
def MaxMS_SEnd(y_sample: Float64Array,
            y_fit: Float64Array,
            tolerance_total: TolerancesInternal
            ) -> float:
    """Intended to clamp the end point within absolute value of tolerance for
    more stability. Returns bigger of:

    - root mean square error
    - end point maximum absolute error
    1. Calculate endpoint maximum absolute error
    2. Calculate residuals squared

    Parameters
    ----------
    y_sample : Float64Array
        Y values of points of data selected for error calculation
    y_fit : Float64Array
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
    """

    residuals = y_fit - y_sample
    # Excess at the last point
    residuals *= residuals
    excess = residuals - tolerance_total
    excess_end = np.amax(excess[-1:])
    excess_mean = np.amax(np.mean(excess, 0))
    return excess_end if excess_end > excess_mean else excess_mean
#%%═════════════════════════════════════════════════════════════════════
def _maxsumabs(residuals: Float64Array, tolerance: float | Float64Array) -> float:
    return np.amax(np.sum(np.abs(residuals) - tolerance) / tolerance)
#───────────────────────────────────────────────────────────────────────

errorfunctions: dict[str, tuple[ErrorFunction, ErrorFunction]] = {
    f.__name__: py_and_nb(f) for f in
    (AbsEnd, MaxAbs, MaxMAbs, MaxMAbs_AbsEnd, MaxMS, MaxMS_SEnd)
    }
