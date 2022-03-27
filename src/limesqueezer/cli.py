'''
Command line interface for processing command line input
'''
#%%═════════════════════════════════════════════════════════════════════
# IMPORT
import numpy as np
import pathlib
import sys
import time
from . import API as ls
from . import reference as ref
helpstring = 'No arguments given'

#───────────────────────────────────────────────────────────────────────
def main():
    '''The main command line app'''
    args = sys.argv[1:]
    is_verbose =  ('--verbose' in args or '-v' in args)
    is_timed = ('--timed' in args or '-t' in args)
    is_plot = '--plot' in args
    is_save = '--save' in args
    is_show = '--show' in args
    use_numba = '--numba' in args
    if len(args) == 0:
        print(helpstring)
        sys.exit()
    path_cwd = pathlib.Path.cwd()

    if is_verbose: print('Selected path is:\n\t%s' % path_cwd)
    if args[0] == 'debug': debug()
    if args[0] == 'timed': timed(int(use_numba))
    if args[0] == 'benchmark': benchmark(int(use_numba))
    if args[0] == 'sandbox': #──────────────────────────────────────────
        args = args[1:]
        import sandbox
    sys.exit()
#%%═════════════════════════════════════════════════════════════════════
# UI UTILITES
def debug():
    x_data, y_data = ref.raw_sine_x2(1e4)
    print(f'{x_data.shape=}\t{y_data.shape=}')
    ls._G['debug'] = True
    xc, yc = ls.compress(x_data, y_data, tol = 1e-2)
    print(f'{len(x_data)=}\t{len(xc)=}')
    time.sleep(1)
    input()
#───────────────────────────────────────────────────────────────────────
def timed(use_numba: int):
    x_data, y_data = ref.raw_sine_x2(1e4)
    print(f'{x_data.shape=}\t{y_data.shape=}')
    ls._G['timed'] = True

    xc, yc = ls.compress(x_data, y_data, tol = 1e-2, use_numba = use_numba)

    print(f'{len(x_data)=}\t{len(xc)=}')
    print(f'runtime {ls._G["runtime"]*1e3:.1f} ms')
#───────────────────────────────────────────────────────────────────────
def benchmark(use_numba: int):
    x_data = np.linspace(0,6,int(1e5))
    y_data = np.array([np.sin(x_data*x_data), np.sin(x_data*1.5*x_data)]).T
    print(f'{x_data.shape=}\t{y_data.shape=}')
    ls._G['timed'] = True
    runtime = 0

    for _ in range(100):
        xc, yc = ls.compress(x_data, y_data, tol = 1e-2,
                            use_numba = use_numba)
        runtime += ls._G["runtime"]
    
    print(f'{len(x_data)=}\t{len(xc)=}')
    print(f'runtime {runtime*1e3:.1f} ms')