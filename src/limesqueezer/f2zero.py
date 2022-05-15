import numpy as np
from .auxiliaries import wait
from . import GLOBALS
global G
G = GLOBALS.dictionary

#───────────────────────────────────────────────────────────────────────
def _get(x, y, x0, y0, tol, sqrtrange, f_fit, errorfunction):
    def f2zero(i: int) -> tuple:
        '''Function such that i is optimal when f2zero(i) = 0'''
        inds = sqrtrange(i)
        residuals, fit = f_fit(x[inds], y[inds], x0, y0)
        return errorfunction(residuals, tol), fit
    return f2zero
#───────────────────────────────────────────────────────────────────────
def _get_debug(x, y, x0, y0, tol, sqrtrange, f_fit, errorfunction):
    def f2zero_debug(i: int) -> tuple:
        '''Function such that i is optimal when f2zero(i) = 0'''
        inds = sqrtrange(i)
        residuals, fit = f_fit(x[inds], y[inds], x0, y0)
        if len(residuals) == 1:
            print(f'\t\t{residuals=}')
        print(f'\t\tstart = {G["start"]} end = {i + G["start"]} points = {i + 1}')
        print(f'\t\tx0\t{x0}\n\t\tx[0]\t{x[inds][0]}\n\t\tx[-1]\t{x[inds][-1]}\n\t\txstart = {G["x"][G["start"]]}')
        indices_all = np.arange(-1, i) + G['start']
        G['x_plot'] = G['x'][indices_all]
        G['y_plot'] = G['interp'](G['x_plot'], x0, x[inds][-1], y0, fit)
        # print(f'{G["y_plot"].shape=}')
        G['line_fit'].set_xdata(G['x_plot'])
        G['line_fit'].set_ydata(G['y_plot'])
        # print(f'{G["y"][indices_all].shape=}')
        res_all = G['y_plot'][1:] - G['y'][indices_all].flatten()[1:]
        print(f'\t\t{residuals.shape=}\n\t\t{res_all.shape=}')
        G['ax_res'].clear()
        G['ax_res'].grid()
        G['ax_res'].axhline(color = 'red', linestyle = '--')
        G['ax_res'].set_ylabel('Residual relative to tolerance')
        G['ax_res'].plot(indices_all[1:] - G['start'], np.abs(res_all) / G['tol']-1,
                            '.', color = 'blue', label = 'ignored')
        G['ax_res'].plot(inds, np.abs(residuals) / G['tol']-1,
                            'o', color = 'red', label = 'sampled')
        G['ax_res'].legend(loc = 'lower right')
        wait('\t\tFitting\n')
        return errorfunction(residuals, tol), fit
    return f2zero_debug
#───────────────────────────────────────────────────────────────────────
def get(*args):
    '''Generates function for the root finder'''
    return _get_debug(*args) if G['debug'] else _get(*args)