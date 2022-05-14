import numpy as np
import numba
#%%═════════════════════════════════════════════════════════════════════
## ERROR TERM
def _maxmaxabs_python(residuals: np.ndarray, tolerances: np.ndarray) -> float:
    '''Python version'''
    residuals = np.abs(residuals)
    # print(f'{residuals.shape=}')
    # print(f'{tolerance.shape=}')
    r_max = residuals[0,0] # Initialising
    # Going through first column to initialise maximum value
    for r0 in residuals[:,0]:
        if r0 > r_max: r_max = r0
    dev_max = r_max - tolerances[0] # Initialise maximum value

    for i, tol in enumerate(tolerances[1:]):
        r_max = residuals[0,i] # Initialise maximum value
        for r0 in residuals[1:,i]:
            if r0 > r_max: r_max = r0
        deviation = r_max - tol
        if deviation > dev_max: dev_max = deviation
    return dev_max
#───────────────────────────────────────────────────────────────────────
@numba.jit(nopython=True, cache=True, fastmath = True)
def _maxmaxabs_numba(residuals: np.ndarray, tolerances: np.ndarray) -> float:
    '''Numba version'''
    residuals = np.abs(residuals)
    # print(f'{residuals.shape=}')
    # print(f'{tolerance.shape=}')
    r_max = residuals[0,0] # Initialising
    # Going through first column to initialise maximum value
    for r0 in residuals[:,0]:
        if r0 > r_max: r_max = r0
    dev_max = r_max - tolerances[0] # Initialise maximum value

    for i, tol in enumerate(tolerances[1:]):
        r_max = residuals[0,i] # Initialise maximum value
        for r0 in residuals[1:,i]:
            if r0 > r_max: r_max = r0
        deviation = r_max - tol
        if deviation > dev_max: dev_max = deviation
    return dev_max
#%%═════════════════════════════════════════════════════════════════════
def _maxRMS_python(residuals: np.ndarray, tolerances: np.ndarray)-> float:
    '''Root mean square error without numba.
    1. Calculate residuals squared
    2. Square root of mean along a column
    3. Find largest of those difference to tolerance'''
    residuals = np.sqrt(np.mean(residuals * residuals, axis = 0))
    dev_max = residuals[0] - tolerances[0] # Initialise maximum value
    for RMS, tol in zip(residuals[1:], tolerances[1:]):
        deviation = RMS - tol
        if deviation > dev_max: dev_max = deviation
    return dev_max
#───────────────────────────────────────────────────────────────────────
@numba.jit(nopython=True, cache=True)
def _maxRMS_numba(residuals: np.ndarray, tolerances: np.ndarray)-> float:
    '''Root mean square error using Numba.
    1. Calculate residuals squared
    2. Square root of mean along a column
    3. Find largest of those difference to tolerance'''
    residuals = np.sqrt(np.mean(residuals * residuals, axis = 0))
    dev_max = residuals[0] - tolerances[0] # Initialise maximum value
    for RMS, tol in zip(residuals[1:], tolerances[1:]):
        deviation = RMS - tol
        if deviation > dev_max: dev_max = deviation
    return dev_max
#%%═════════════════════════════════════════════════════════════════════
def _maxRMS_absend_python(residuals: np.ndarray, tolerances: np.ndarray)-> float:
    '''Without Numba. Intended to clamp the end point within absolute value of tolerance for more stability. Returns bigger of:
    - root mean square error
    - end point maximum absolute error
    1. Calculate endpoint maximum absolute error
    2. Calculate residuals squared
    3. Square root of mean along a column
    4. Find largest of those difference to tolerance
    5. Calculate absolute error of the end point
    6. Return the sum of end point error and RMS error'''
    # End point absolute error
    absend_max = np.max(np.abs(residuals[-1:]) - tolerances)
    residuals = np.sqrt(np.mean(residuals * residuals, axis = 0))
    dev_max_RMS = residuals[0] - tolerances[0] # Initialise maximum value
    for RMS, tol in zip(residuals[1:], tolerances[1:]):
        deviation = RMS - tol
        if deviation > dev_max_RMS: dev_max_RMS = deviation
    
    return max(absend_max, dev_max_RMS)
#───────────────────────────────────────────────────────────────────────
@numba.jit(nopython=True, cache=True)
def _maxRMS_absend_numba(residuals: np.ndarray, tolerances: np.ndarray)-> float:
    '''With Numba. Intended to clamp the end point within absolute value of tolerance for more stability. Returns bigger of:
    - root mean square error
    - end point maximum absolute error
    1. Calculate endpoint maximum absolute error
    2. Calculate residuals squared
    3. Square root of mean along a column
    4. Find largest of those difference to tolerance
    5. Calculate absolute error of the end point
    6. Return the sum of end point error and RMS error'''
    # End point absolute error
    absend_max = np.max(np.abs(residuals[-1:]) - tolerances)
    residuals = np.sqrt(np.mean(residuals * residuals, axis = 0))
    dev_max_RMS = residuals[0] - tolerances[0] # Initialise maximum value
    for RMS, tol in zip(residuals[1:], tolerances[1:]):
        deviation = RMS - tol
        if deviation > dev_max_RMS: dev_max_RMS = deviation
    
    return max(absend_max, dev_max_RMS)
#%%═════════════════════════════════════════════════════════════════════
def maxsumabs(residuals: np.ndarray,tolerance: np.ndarray) -> float:
    return np.amax(np.sum(np.abs(residuals) - tolerance) / tolerance)
#───────────────────────────────────────────────────────────────────────
#%%═════════════════════════════════════════════════════════════════════
dictionary = {'maxmaxabs': (_maxmaxabs_python, _maxmaxabs_numba),
              'maxRMS': (_maxRMS_python, _maxRMS_numba),
              'maxRMS_absend': (_maxRMS_absend_python, _maxRMS_absend_numba)}
#───────────────────────────────────────────────────────────────────────
def get(name, use_numba):
    try:
        versions = dictionary[name]
    except KeyError:
        raise ValueError(f'Error function {name} not recognised. Valid are {tuple(dictionary.keys())}')
    try:
        return versions[use_numba]
    except IndexError:
        raise ValueError(f'Numba selector should be 0 or 1, not {use_numba}')