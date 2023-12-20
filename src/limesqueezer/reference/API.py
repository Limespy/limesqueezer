import math
import pathlib
from collections import namedtuple

import numpy as np

from .. import _API as ls # Careful with this circular import
#%%═════════════════════════════════════════════════════════════════════
# SETUP

path_data = pathlib.Path(__file__).parent.absolute() / 'data'

Reference = namedtuple('Reference',
                       ['raw','atol','cmethod','ostyle','compressed'],
                       defaults=[1e-5,'interp10','monolith',None])

#%%═════════════════════════════════════════════════════════════════════
# FUNCTIONS


##%%════════════════════════════════════════════════════════════════════
## Auxiliary functions

##%%═════════════════════════════════════════════════════════════════════
## Reference functions
def f2zero_100(n: int) -> tuple[float, bool]:
    """Returns < 0 for values 0 to 100 and >0 for values > 100."""
    if round(n) != n: raise ValueError('Not whole number')
    if n < 0: raise ValueError('Input must be >= 0')
    return np.sqrt(n) - 10.01, True
#%%═════════════════════════════════════════════════════════════════════
# Reference raw data
def raw_poly0(n = 1e1):
    x = np.linspace(0, 1, int(n))
    return x, np.zeros(len(x))
#───────────────────────────────────────────────────────────────────────
def raw_poly1(n = 1e1):
    x = np.linspace(0, 1, int(n))
    return x, x
#───────────────────────────────────────────────────────────────────────
def raw_poly2(n = 1e2):
    x = np.linspace(0, 1, int(n))
    return x, np.array(x ** 2)
#───────────────────────────────────────────────────────────────────────
def raw_sine(n = 1e4):
    x = np.linspace(0, 6, int(n))
    return x, np.array(np.sin(x * 2 * math.pi))
#───────────────────────────────────────────────────────────────────────
def raw_sine_x2(n = 1e4):
    x = np.linspace(0, 6, int(n))
    return x, np.array(np.sin(x * x))
#───────────────────────────────────────────────────────────────────────
def raw_sine_x2_normal(n = 1e4, std = 0.1):
    rng = np.random.default_rng(12345)
    x = np.linspace(0, 6, int(n))
    return x, np.array(np.sin(x * x)) + std * rng.standard_normal(int(n))
#───────────────────────────────────────────────────────────────────────
raw = {'poly0': raw_poly0,
       'poly1': raw_poly1,
       'poly2': raw_poly2,
       'sine': raw_sine}
