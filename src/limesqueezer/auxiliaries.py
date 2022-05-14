import numpy as np
import math
import sys
import numba
#%%═════════════════════════════════════════════════════════════════════
# AUXILIARIES
def to_ndarray(item, shape = ()) :
    if not hasattr(item, '__iter__'): # Not some iterable
        if -1 in shape: # Array of shape length of dimensions with one item
            return np.array(item, ndmin = len(shape))
        else:
            return np.full(shape, item) # Array of copies in the shape
    elif not isinstance(item, np.ndarray): # Iterable into array
        item = np.array(item)
    return item if shape == () else item.reshape(shape)
#───────────────────────────────────────────────────────────────────────
def sqrtrange_python(n: int):
    '''~ sqrt(n + 2) equally spaced integers including the n'''
    inds = np.arange(0, n + 1, round((n + 1) ** 0.5) )
    inds[-1] = n
    return inds
#───────────────────────────────────────────────────────────────────────
@numba.jit(nopython = True, cache = True, fastmath = True)
def sqrtrange_numba(n: int):
    '''~ sqrt(n + 2) equally spaced integers including the n'''
    inds = np.arange(0, n + 1, round((n + 1) ** 0.5) )
    inds[-1] = n
    return inds
#───────────────────────────────────────────────────────────────────────
def wait(text = ''):
    if input(text) in ('e', 'q', 'exit', 'quit'): sys.exit()
#───────────────────────────────────────────────────────────────────────
def stats(x_data, xc):
    # What if the data was compressed by sampling at the minimum interval of the compressed
    datarange = x_data[-1] - x_data[0]
    minslice = np.min(np.diff(xc))
    maxslice = np.max(np.diff(xc))
    # ameanslice = (minslice + maxslice) / 2
    # gmeanslice = (minslice * maxslice)**0.5
    hmeanslice = 2 / (1 / minslice + 1 / maxslice)

    return f'''{len(x_data) / len(xc):.0f} compression ratio
{datarange / hmeanslice / len(xc):.1f} x better than mean slices
{datarange / minslice / len(xc):.1f} x better than minimum slices'''
