import pathlib
import numpy as np
from collections import namedtuple
path_package = pathlib.Path(__file__).parent.absolute()

import sys
sys.path.insert(1,str(path_package.parent))
import API as compression
#%%═════════════════════════════════════════════════════════════════════
# SETUP

path_data = path_package / 'data'

Reference = namedtuple('Reference', 
                       ['raw','atol','cmethod','ostyle','compressed'],
                       defaults=[1e-5,'interp10','monolith',None])

#%%═════════════════════════════════════════════════════════════════════
# FUNCTIONS


##%%════════════════════════════════════════════════════════════════════
## Auxiliary functions


#%%═════════════════════════════════════════════════════════════════════
# Reference raw data
def raw_poly0(n=1e1):
    x = np.linspace(0,1,int(n))
    return x, np.zeros(len(x))
#───────────────────────────────────────────────────────────────────────
def raw_poly1(n=1e1):
    x = np.linspace(0,1,int(n))
    return x, x
#───────────────────────────────────────────────────────────────────────
def raw_poly2(n=1e2):
    x = np.linspace(0,1,int(n))
    return x, np.array([x**2,2*x**2])
#───────────────────────────────────────────────────────────────────────
raw = {'poly0': raw_poly0,
       'poly1': raw_poly1,
       'poly2': raw_poly2}

#───────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────
references = [Reference(raw['poly0'],1e-5,'interp10','monolith')]
#───────────────────────────────────────────────────────────────────────
def generate(function, method, ytol=1e-3):
    x_ref, y_ref = raw[function]()
    xc, yc, _ = compression.compress(x_ref, y_ref, method=method, ytol=[ytol,2*ytol])
    print('xc',xc)
    print('xc reshape', xc.reshape([-1,1]))
    print(yc)

    np.savetxt(path_data / (function+'_'+method+'.csv'),
               np.concatenate((xc.reshape([-1,1]), yc), axis=1), delimiter=',')
