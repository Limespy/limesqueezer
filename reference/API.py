import pathlib
import numpy as np
from collections import namedtuple


#%%═════════════════════════════════════════════════════════════════════
# SETUP

path_data = pathlib.Path(__file__).absolute().parent() / 'data'

Reference = namedtuple('Reference', 
                       ['raw','atol','cmethod','ostyle','compressed'],
                       defaults=[1e-5,'interp10','monolith',None])

#%%═════════════════════════════════════════════════════════════════════
# FUNCTIONS


##%%════════════════════════════════════════════════════════════════════
## Auxiliary functions


#%%═════════════════════════════════════════════════════════════════════
# Reference raw data
def raw_0(x):
    x = np.linspace(0,1,int(1e1))
    return x, np.zeros(len(x))
#───────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────
references = [Reference(raw_0,1e-5,'interp10','monolith')]