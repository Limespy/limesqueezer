__version__ = '1.0.6'

import sys
import pathlib

sys.path.insert(1,str(pathlib.Path(__file__).parent.absolute()))

from .API import *

helpstring = 'No arguments given'

if len(sys.argv)==1:
    print(helpstring)
    exit()
elif sys.argv[1] == 'debug':
    import plotters
    debugplot = plotters.Debug()
    debugplot.run()
    exit()