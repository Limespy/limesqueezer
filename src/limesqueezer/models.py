from .auxiliaries import Callable, FloatArray, MaybeArray, default_numba_kwargs, py_and_nb
import numpy as np
import numba as nb
import collections

FitFunction = Callable[[FloatArray, FloatArray, float, FloatArray], FloatArray]
Interpolator = Callable[[MaybeArray, float, float, FloatArray, FloatArray], FloatArray]
#%%═════════════════════════════════════════════════════════════════════
# BUILTIN COMPRESSION MODELS
FitSet = collections.namedtuple('FitSet', ('fit', 'interpolate'))
#───────────────────────────────────────────────────────────────────────
def _fit_Poly10(x: FloatArray,
                y: FloatArray,
                x0:float,
                y0: MaybeArray,
                ) -> FloatArray:
    '''Takes block of data, previous fitting parameters and calculates next fitting parameters

    Parameters
    ----------
    x : FloatArray
        x values of the points to be fitted
    y : FloatArray
        y values of the points to be fitted
    x0 : float
        Last compressed point x value
    y0 : FloatArray
        Last compressed point y value(s)

    Returns
    -------
    FloatArray
        Fitted y-values
    '''

    Dx: FloatArray = x - x0
    Dy: FloatArray = y - y0
    return np.outer(Dx, Dx @ Dy / Dx.dot(Dx)) + y0
#───────────────────────────────────────────────────────────────────────
def _interp_Poly10(x: MaybeArray,
                   x1: float, x2:float,
                   y1: FloatArray, y2: FloatArray
                     ) -> FloatArray:
        '''Interpolates between two consecutive points of compressed data'''
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1
#───────────────────────────────────────────────────────────────────────
#═══════════════════════════════════════════════════════════════════════
dictionary = {'Poly10':
                    FitSet(py_and_nb(_fit_Poly10), py_and_nb(_interp_Poly10))}
#───────────────────────────────────────────────────────────────────────
def get(name: str) -> FitSet:
    '''Access fitting model while doing errorhandling

    Parameters
    ----------
    name : str
        name of the model to be accesses

    Returns
    -------
        Fit NamedTuple

    Raises
    ------
    NotImplementedError
        If name in the dictionary of implemented builtin models
    '''
    try:
        return dictionary[name]
    except KeyError:
        raise NotImplementedError(f'Builtin fit function {name} not recognised. Valid are {tuple(dictionary.keys())}')

