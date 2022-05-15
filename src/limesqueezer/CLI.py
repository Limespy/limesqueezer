'''
Command line interface for processing command line input
'''
#%%═════════════════════════════════════════════════════════════════════
# IMPORT
import numpy as np
import pathlib
import sys
import time
from .GLOBALS import dictionary as G
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
    G['timed'] = '--timed' in args
    G['debug'] = '--debug' in args
    use_numba = int('--numba' in args)
    if len(args) == 0:
        print(helpstring)
        sys.exit()
    path_cwd = pathlib.Path.cwd()

    if is_verbose: print('Selected path is:\n\t%s' % path_cwd)
    if args[0] == 'sandbox': #──────────────────────────────────────────
        args = args[1:]
        import sandbox
    else:
        run(args, use_numba, is_plot, is_timed)
    sys.exit()
#%%═════════════════════════════════════════════════════════════════════
# UI UTILITES
def run(args: list, use_numba: int, is_plot: bool, is_timed: bool):
    x_data, y_data = ref.raw_sine_x2_normal(1e4, std=0.0001)
    # y_data[1000] += 1 
    if args[0] == 'block':
        xc, yc = ls.compress(x_data, y_data, tol = 1e-2,
                    use_numba = use_numba, errorfunction = 'maxmaxabs', fitset = 'Poly10')
        print(ls.aux.stats(x_data, xc))
    elif args[0] == 'stream':
        xc, yc = _stream(x_data, y_data, 1e-2, use_numba)
    elif args[0] == 'both':
        xcb, ycb = ls.compress(x_data, y_data, tol = 1e-2, use_numba = use_numba, initial_step = 100, errorfunction = 'maxmaxabs')
        xcs, ycs = _stream(x_data, y_data, 1e-2, use_numba)
        for i, (xb, xs) in enumerate(zip(xcb,xcs)):
            if xb != xs:
                print(f'Deviation at {i=}, {xb=}, {xs=}')
                break
        for i, (xb, xs) in enumerate(zip(reversed(xcb),reversed(xcs))):
            if xb != xs:
                print(f'Deviation at {i=}, {xb=}, {xs=}')
                break
        print(xcb)
        print(xcs)
        
        xc, yc = xcb, ycb
    if is_plot:
        from . import plotters
        plotters.comparison(x_data, y_data, ls.decompress(xc, yc)(x_data))
    # print(f'{xc[-10:-1]}')
    print(f'{len(x_data)=}\t{len(xc)=}')
    if is_timed: print(f'runtime {G["runtime"]*1e3:.1f} ms')
#───────────────────────────────────────────────────────────────────────
def _stream(x_data: np.ndarray, y_data: np.ndarray, tol: float, use_numba: int):

    with ls.Stream(x_data[0], y_data[0], tol = tol, use_numba = use_numba) as record:
        for x, y in zip(x_data[1:], y_data[1:]):
            record(x, y)
    return record.x, record.y
