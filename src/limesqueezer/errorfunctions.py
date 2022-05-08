import numpy as np
import numba
#%%═════════════════════════════════════════════════════════════════════
## ERROR TERM
def _maxmaxabs_python(residuals: np.ndarray, tolerance: np.ndarray) -> float:
    '''Python version'''
    residuals = np.abs(residuals)
    # print(f'{residuals.shape=}')
    # print(f'{tolerance.shape=}')
    r_max = residuals[0,0] # Initialising

    # Going through first column to initialise maximum value
    for r0 in residuals[:,0]:
        if r0 > r_max: r_max = r0
    dev_max = r_max - tolerance[0]

    for i, k in enumerate(tolerance[1:]):
        for r0 in residuals[:,i]:
            if r0 > r_max: r_max = r0
        deviation = r_max - k
        if deviation > dev_max: dev_max = deviation
    return dev_max
#───────────────────────────────────────────────────────────────────────
@numba.jit(nopython=True, cache=True, fastmath = True)
def _maxmaxabs_numba(residuals: np.ndarray, tolerance: np.ndarray) -> float:
    residuals = np.abs(residuals)
    # print(f'{residuals.shape=}')
    # print(f'{tolerance.shape=}')
    r_max = residuals[0,0] # Initialising

    # Going through first column to initialise maximum value
    for r0 in residuals[:,0]:
        if r0 > r_max: r_max = r0
    dev_max = r_max - tolerance[0]

    for i, k in enumerate(tolerance[1:]):
        for r0 in residuals[:,i]:
            if r0 > r_max: r_max = r0
        deviation = r_max - k
        if deviation > dev_max: dev_max = deviation
    return dev_max
#───────────────────────────────────────────────────────────────────────
maxmaxabs = (_maxmaxabs_python, _maxmaxabs_numba)
#%%═════════════════════════════════════════════════════════════════════
def _maxRMS_python(residuals: np.ndarray, tolerance: np.ndarray)-> float:
    residuals *= residuals
    # print(f'{residuals.shape=}')
    # print(f'{tolerance.shape=}')
    m = np.sqrt(np.mean(residuals[:,0])) - tolerance[0]
    for i, k in enumerate(tolerance[1:]):
        m = max(m, np.sqrt(np.mean(residuals[:,i])) - k)
    return m
#───────────────────────────────────────────────────────────────────────
@numba.jit(nopython=True, cache=True)
def _maxRMS_numba(residuals: np.ndarray, tolerance: np.ndarray)-> float:
    residuals *= residuals
    # print(f'{residuals.shape=}')
    # print(f'{tolerance.shape=}')
    m = np.sqrt(np.mean(residuals[:,0])) - tolerance[0]
    for i, k in enumerate(tolerance[1:]):
        m = max(m, np.sqrt(np.mean(residuals[:,i])) - k)
    return m
#───────────────────────────────────────────────────────────────────────
maxRMS = (_maxRMS_python, _maxRMS_numba)
#%%═════════════════════════════════════════════════════════════════════
def maxsumabs(residuals: np.ndarray,tolerance: np.ndarray) -> float:
    return np.amax(np.sum(np.abs(residuals) - tolerance) / tolerance)
#───────────────────────────────────────────────────────────────────────
#%%═════════════════════════════════════════════════════════════════════
dictionary = {'maxmaxabs': maxmaxabs,
              'maxRMS': maxRMS}
#───────────────────────────────────────────────────────────────────────
def get_errorfunction(name, use_numba, tol):
    _errorfunction = dictionary[name][use_numba]
    return lambda residuals: _errorfunction(residuals, tol)