"""Functions to be solved in discrete root finding"""
from .auxiliaries import wait, SqrtRange
from .GLOBALS import (G, 
                      FloatArray,
                      Callable,
                      TolerancesInternal)
from .models import  FitFunction
from .errorfunctions import ErrorFunction
import numpy as np
#───────────────────────────────────────────────────────────────────────
F2Zero = Callable[[int], tuple[float, FloatArray]]
Get =  Callable[[FloatArray,
                       FloatArray,
                       float,
                       FloatArray,
                       TolerancesInternal,
                       SqrtRange,
                       FitFunction,
                       ErrorFunction],
                      F2Zero]
def get(x: FloatArray,
        y: FloatArray,
        x0: float,
        y0: FloatArray,
        tol: TolerancesInternal,
        sqrtrange: SqrtRange,
        f_fit: FitFunction,
        errorfunction: ErrorFunction
        ) -> F2Zero:
    def f2zero(i: int) -> tuple[float, FloatArray]:
        '''Function such that i is optimal when f2zero(i) = 0

        Parameters
        ----------
        i : int
            highest index of the fit 

        Returns
        -------
        tuple
            output of the error function and last of the fit
        '''
        inds = sqrtrange(i)
        x_sample, y_sample = x[inds], y[inds]
        y_fit = f_fit(x_sample, y_sample, x0, y0)
        return errorfunction(y_sample, y_fit, tol), y_fit[-1]
    return f2zero
#───────────────────────────────────────────────────────────────────────
def get_debug(x: FloatArray,
              y: FloatArray,
              x0: float,
              y0: FloatArray,
              tol: TolerancesInternal,
              sqrtrange: Callable[[int], np.int64],
              f_fit: FitFunction,
              errorfunction: ErrorFunction
              ) -> Callable[[int], tuple[float, FloatArray]]:
    def f2zero(i: int) -> tuple[float, FloatArray]:
        '''Function such that i is optimal when f2zero(i) = 0'''
        inds = sqrtrange(i)
        x_sample, y_sample = x[inds], y[inds]
        y_fit = f_fit(x_sample, y_sample, x0, y0)
        residuals = y_fit - y_sample
        if len(residuals) == 1:
            print(f'\t\t{residuals=}')
        print(f'\t\tstart = {G["start"]} end = {i + G["start"]} points = {i + 1}')
        print(f'\t\tx0\t{x0}\n\t\tx[0]\t{x[inds][0]}\n\t\tx[-1]\t{x[inds][-1]}\n\t\txstart = {G["x"][G["start"]]}')
        indices_all = np.arange(-1, i) + G['start']
        G['x_plot'] = G['x'][indices_all]
        G['y_plot'] = G['interp'](G['x_plot'], x0, x[inds][-1], y0, y_fit[-1])
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
        indices_x = indices_all[1:] - G['start']
        residuals_relative =  np.abs(res_all) / G['tol'] - 1
        G['ax_res'].plot(indices_x, residuals_relative,
                            '.', color = 'blue', label = 'ignored')
        G['ax_res'].plot(inds, np.abs(residuals) / G['tol']-1,
                            'o', color = 'red', label = 'sampled')
        G['ax_res'].legend(loc = 'lower right')
        wait('\t\tFitting\n')
        return errorfunction(y_sample, y_fit, tol), y_fit[-1]
    return f2zero
