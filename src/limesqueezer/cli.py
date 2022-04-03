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
    ls._G['timed'] = '--timed' in args
    ls._G['debug'] = '--debug' in args
    use_numba = int('--numba' in args)
    if len(args) == 0:
        print(helpstring)
        sys.exit()
    path_cwd = pathlib.Path.cwd()

    if is_verbose: print('Selected path is:\n\t%s' % path_cwd)
    if args[0] == 'sandbox': #──────────────────────────────────────────
        args = args[1:]
        import sandbox
    elif args[0] == 'benchmark': #──────────────────────────────────────
        import benchmark
    else:
        run(args, use_numba)
    sys.exit()
#%%═════════════════════════════════════════════════════════════════════
# UI UTILITES
def run(args, use_numba: int):
    xdata, ydata = ref.raw_sine_x2(1e4)
    if args[0] == 'block':
        xc, yc = ls(xdata, ydata, tol = 1e-2,
                    use_numba = use_numba, errorfunction = 'maxmaxabs')
    elif args[0] == 'stream':
        xc, yc = _stream(xdata, ydata, 1e-2, use_numba)
    elif args[0] == 'both':
        xcb, ycb = ls(xdata, ydata, tol = 1e-2, use_numba = use_numba, initial_step = 100, errorfunction = 'maxmaxabs')
        xcs, ycs = _stream(xdata, ydata, 1e-2, use_numba)
        for i, (xb, xs) in enumerate(zip(xcb,xcs)):
            if xb - xs != 0:
                print(f'{i=}, {xb=}, {xs=}')
                break
        for i, (xb, xs) in enumerate(zip(reversed(xcb),reversed(xcs))):
            if xb - xs != 0:
                print(f'{i=}, {xb=}, {xs=}')
                break
        xc = xcb
    # print(f'{xc[-10:-1]}')
    print(f'{len(xdata)=}\t{len(xc)=}')
    if ls._G['timed']: print(f'runtime {ls._G["runtime"]*1e3:.1f} ms')
#───────────────────────────────────────────────────────────────────────
def _stream(xdata: np.ndarray, ydata: np.ndarray, tol: float, use_numba: int):

    with ls.Stream(xdata[0], ydata[0], tol = tol, use_numba = use_numba) as record:
        for x, y in zip(xdata[1:], ydata[1:]):
            record(x, y)
    return record.x, record.y
