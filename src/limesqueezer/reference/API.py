import sys
import pathlib
import numpy as np
from collections import namedtuple
from scipy import interpolate
import math

from .. import API as ls # Careful with this circular import
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
def f2zero_100(n: int) -> float:
    '''returns < 0 for values 0 to 100 and >0 for values > 100'''
    if round(n) != n: raise ValueError('Not whole number')
    if n < 0: raise ValueError('Input must be >= 0')
    return np.sqrt(n) - 10.01, True
#%%═════════════════════════════════════════════════════════════════════
# Reference raw data
def raw_poly0(n = 1e1):
    x = np.linspace(0,1,int(n))
    return x, np.zeros(len(x))
#───────────────────────────────────────────────────────────────────────
def raw_poly1(n = 1e1):
    x = np.linspace(0,1,int(n))
    return x, x
#───────────────────────────────────────────────────────────────────────
def raw_poly2(n = 1e2):
    x = np.linspace(0,1,int(n))
    return x, np.array(x**2)
#───────────────────────────────────────────────────────────────────────
def raw_sine(n = 1e4):
    x = np.linspace(0,6,int(n))
    return x, np.array(np.sin(x*2*math.pi))
#───────────────────────────────────────────────────────────────────────
def raw_sine_x2(n = 1e4):
    x = np.linspace(0,6,int(n))
    return x, np.array(np.sin(x*x))
#───────────────────────────────────────────────────────────────────────
def raw_sine_normal(n = 1e4, std=0.1):
    rng = np.random.default_rng(12345)
    x = np.linspace(0,1,int(n))
    return x, np.array(np.sin(x*2*math.pi)) + std*rng.standard_normal(int(n))
#───────────────────────────────────────────────────────────────────────
raw = {'poly0': raw_poly0,
       'poly1': raw_poly1,
       'poly2': raw_poly2,
       'sine': raw_sine}
###═════════════════════════════════════════════════════════════════════
class Data():
    '''Data container'''
    def __init__(self, function, ytol=1e-2):
        
        self.x, self.y = raw[function]()
        self.y_range = np.max(self.y) - np.min(self.y)
        self.ytol = ytol
        self.xc = None
        self.yc = None
    #───────────────────────────────────────────────────────────────────
    def make_lerp(self):
        self.lerp = interpolate.interp1d(self.xc.flatten(), self.yc.flatten(),
                                            assume_sorted=True)
        self.residuals = self.lerp(self.x) - self.y
        self.residuals_relative = self.residuals / self.ytol
        self.residuals_relative_cumulative = np.cumsum(self.residuals_relative)
        self.NRMSE = np.std(self.residuals)/self.y_range
        self.covariance = np.cov((self.lerp(self.x), self.y))
#───────────────────────────────────────────────────────────────────────
references = [Reference(raw['poly0'],1e-5,'interp10','monolith')]
#───────────────────────────────────────────────────────────────────────
def generate(function, method, ytol=5e-2):
    data = Data(function, ytol=ytol)
    data.xc, data.yc, _ = ls.compress(data.x, data.y, method=method, ytol=data.ytol)
    data.make_lerp()
    print(np.amax(np.abs(data.residuals_relative)))
    np.savetxt(path_data / (function+'_'+method+'.csv'),
               np.concatenate((data.xc, data.yc), axis=1), delimiter=',', header='hmm')
    return data
