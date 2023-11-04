import collections

import numpy as np

from .auxiliaries import Callable
from .auxiliaries import Float64Array
from .auxiliaries import MaybeArray
from .auxiliaries import py_and_nb

FitFunction = Callable[[Float64Array, Float64Array, float, Float64Array], Float64Array]
Interpolator = Callable[[MaybeArray, float, float, Float64Array, Float64Array], Float64Array]
#%%═════════════════════════════════════════════════════════════════════
# BUILTIN COMPRESSION MODELS
FitSet = collections.namedtuple('FitSet', ('fit', 'interpolate'))
#───────────────────────────────────────────────────────────────────────
def _fit_Poly10(x: Float64Array,
                y: Float64Array,
                x0:float,
                y0: MaybeArray,
                ) -> Float64Array:
    """Takes block of data, previous fitting parameters and calculates next
    fitting parameters.

    Parameters
    ----------
    x : Float64Array
        x values of the points to be fitted
    y : Float64Array
        y values of the points to be fitted
    x0 : float
        Last compressed point x value
    y0 : Float64Array
        Last compressed point y value(s)

    Returns
    -------
    Float64Array
        Fitted y-values
    """

    Dx: Float64Array = x - x0
    Dy: Float64Array = y - y0
    return np.outer(Dx, Dx @ Dy / Dx.dot(Dx)) + y0
#───────────────────────────────────────────────────────────────────────
def _interp_Poly10(x: MaybeArray,
                   x1: float, x2:float,
                   y1: Float64Array, y2: Float64Array
                     ) -> Float64Array:
        """Interpolates between two consecutive points of compressed data."""
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1
#───────────────────────────────────────────────────────────────────────
Poly10 = FitSet(py_and_nb(_fit_Poly10), py_and_nb(_interp_Poly10))
