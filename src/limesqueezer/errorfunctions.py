import numpy as np
import numba as nb
#%%═════════════════════════════════════════════════════════════════════
## ERROR TERM
def _maxmaxabs_python(y_sample: np.ndarray,
                      y_fit: np.ndarray,
                      tolerances: tuple[np.ndarray, np.ndarray, np.ndarray]
                      ) -> float:
    '''Maximum of absolute orrors relative to tolerance

    Parameters
    ----------
    y_sample : np.ndarray
        Y values of points of data selected for error calculation
    y_fit : np.ndarray
        Y values from fitting the model into data
    tolerances : tuple[np.ndarray, np.ndarray, np.ndarray]
        Tolerances for errors
        1) Relative error array
        2) Absolute error array
        3) Falloff array

    Returns
    -------
    float
        Error value. Should be <0 for fit to be acceptable
    '''
    y_sample_abs = np.abs(y_sample)
    reltols = y_sample_abs * tolerances[0]
    abstols = tolerances[1] / (tolerances[2] * y_sample_abs + 1)
    return np.max(np.abs(y_fit - y_sample) - reltols - abstols)
#───────────────────────────────────────────────────────────────────────
@nb.jit(nopython = True, cache = True, fastmath = True)
def _maxmaxabs_numba(y_sample: np.ndarray,
                     y_fit: np.ndarray,
                     tolerances: tuple[np.ndarray, np.ndarray, np.ndarray]
                     ) -> float:
    '''Maximum of absolute orrors relative to tolerance. Numba version

    Parameters
    ----------
    y_sample : np.ndarray
        Y values of points of data selected for error calculation
    y_fit : np.ndarray
        Y values from fitting the model into data
    tolerances : tuple[np.ndarray, np.ndarray, np.ndarray]
        Tolerances for errors
        1) Relative error array
        2) Absolute error array
        3) Falloff array

    Returns
    -------
    float
        Error value. Should be <0 for fit to be acceptable
    '''
    y_sample_abs = np.abs(y_sample)
    reltols = y_sample_abs * tolerances[0]
    abstols = tolerances[1] / (tolerances[2] * y_sample_abs + 1)
    return np.max(np.abs(y_fit - y_sample) - reltols - abstols)
#%%═════════════════════════════════════════════════════════════════════
def _maxMS_python(y_sample: np.ndarray,
                   y_fit: np.ndarray,
                   tolerances: tuple[np.ndarray, np.ndarray, np.ndarray]
                   ) -> float:
    '''Root mean square error without numba.
    1. Calculate residuals squared
    2. Square root of mean along a column
    3. Find largest of those difference to tolerance

    Parameters
    ----------
    y_sample : np.ndarray
        Y values of points of data selected for error calculation
    y_fit : np.ndarray
        Y values from fitting the model into data
    tolerances : tuple[np.ndarray, np.ndarray, np.ndarray]
        Tolerances for errors
        1) Relative error array
        2) Absolute error array
        3) Falloff array

    Returns
    -------
    float
        Error value. Should be <0 for fit to be acceptable
    '''
    y_sample_abs = np.abs(y_sample)
    reltols = y_sample_abs * tolerances[0]
    abstols = tolerances[1] / (tolerances[2] * y_sample_abs + 1)
    residuals = y_fit - y_sample
    return np.amax(np.mean(residuals * residuals - reltols - abstols, 0))
#───────────────────────────────────────────────────────────────────────
@nb.jit(nopython=True, cache=True, fastmath = True)
def _maxMS_numba(y_sample: np.ndarray,
                  y_fit: np.ndarray,
                  tolerances: tuple[np.ndarray, np.ndarray, np.ndarray]
                  ) -> float:
    '''Root mean square error using Numba.
    1. Calculate residuals squared
    2. Square root of mean along a column
    3. Find largest of those difference to tolerance

    Parameters
    ----------
    y_sample : np.ndarray
        Y values of points of data selected for error calculation
    y_fit : np.ndarray
        Y values from fitting the model into data
    tolerances : tuple[np.ndarray, np.ndarray, np.ndarray]
        Tolerances for errors
        1) Relative error array
        2) Absolute error array
        3) Falloff array

    Returns
    -------
    float
        Error value. Should be <0 for fit to be acceptable
    '''
    y_sample_abs = np.abs(y_sample)
    reltols = y_sample_abs * tolerances[0]
    abstols = tolerances[1] / (tolerances[2] * y_sample_abs + 1)
    residuals = y_fit - y_sample
    return np.amax(np.mean(residuals * residuals - reltols - abstols, 0))
#%%═════════════════════════════════════════════════════════════════════
def _maxMS_absend_python(y_sample: np.ndarray,
                          y_fit: np.ndarray,
                          tolerances: tuple[np.ndarray, np.ndarray, np.ndarray]
                          ) -> float:
    '''Without Numba. Intended to clamp the end point within absolute value of tolerance for more stability. Returns bigger of:
    - root mean square error
    - end point maximum absolute error
    1. Calculate endpoint maximum absolute error
    2. Calculate residuals squared

    Parameters
    ----------
    y_sample : np.ndarray
        Y values of points of data selected for error calculation
    y_fit : np.ndarray
        Y values from fitting the model into data
    tolerances : tuple[np.ndarray, np.ndarray, np.ndarray]
        Tolerances for errors
        1) Relative error array
        2) Absolute error array
        3) Falloff array

    Returns
    -------
    float
        Error value. Should be <0 for fit to be acceptable
    '''
    y_sample_abs = np.abs(y_sample)
    reltols = y_sample_abs * tolerances[0]
    abstols = tolerances[1] / (tolerances[2] * y_sample_abs + 1)
    tols = reltols + abstols

    residuals = y_fit - y_sample
    # Excess at the last point
    residuals *= residuals
    excess_end = np.max(np.abs(residuals[-1:]) - tols[-1])

    excess_max_MS = np.amax(np.mean(residuals - tols, 0))
    return excess_end if excess_end > excess_max_MS else excess_max_MS
#───────────────────────────────────────────────────────────────────────
@nb.jit(nopython=True, cache=True, fastmath = True)
def _maxMS_absend_numba(y_sample: np.ndarray,
                         y_fit: np.ndarray,
                         tolerances: tuple[np.ndarray, np.ndarray, np.ndarray]
                         ) -> float:
    '''With Numba. Intended to clamp the end point within absolute value of tolerance for more stability. Returns bigger of:
    - root mean square error
    - end point maximum absolute error
    1. Calculate endpoint maximum absolute error
    2. Calculate residuals squared

    Parameters
    ----------
    y_sample : np.ndarray
        Y values of points of data selected for error calculation
    y_fit : np.ndarray
        Y values from fitting the model into data
    tolerances : tuple[np.ndarray, np.ndarray, np.ndarray]
        Tolerances for errors
        1) Relative error array
        2) Absolute error array
        3) Falloff array

    Returns
    -------
    float
        Error value. Should be <0 for fit to be acceptable
    '''
    y_sample_abs = np.abs(y_sample)
    reltols = y_sample_abs * tolerances[0]
    abstols = tolerances[1] / (tolerances[2] * y_sample_abs + 1)
    tols = reltols + abstols

    residuals = y_fit - y_sample
    # Excess at the last point
    residuals *= residuals
    excess_end = np.max(np.abs(residuals[-1:]) - tols[-1])

    excess_max_MS = np.amax(np.mean(residuals - tols, 0))
    return excess_end if excess_end > excess_max_MS else excess_max_MS
#%%═════════════════════════════════════════════════════════════════════
def maxsumabs(residuals: np.ndarray, tolerance: np.ndarray) -> float:
    return np.amax(np.sum(np.abs(residuals) - tolerance) / tolerance)
#───────────────────────────────────────────────────────────────────────
#%%═════════════════════════════════════════════════════════════════════
dictionary = {'maxmaxabs': (_maxmaxabs_python, _maxmaxabs_numba),
              'maxMS': (_maxMS_python, _maxMS_numba),
              'maxMS_absend': (_maxMS_absend_python, _maxMS_absend_numba)}
#───────────────────────────────────────────────────────────────────────
def get(name, use_numba):
    try:
        versions = dictionary[name]
    except KeyError:
        raise NotImplementedError(f'Error function {name} not recognised. Valid are {tuple(dictionary.keys())}')
    try:
        return versions[use_numba]
    except IndexError:
        raise ValueError(f'Numba selector should be 0 or 1, not {use_numba}')