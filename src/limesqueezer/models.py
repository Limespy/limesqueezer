import numpy as np
import numba
#%%═════════════════════════════════════════════════════════════════════
# BUILTIN COMPRESSION MODELS
class Poly10:
    """Builtin group of functions for doing the compression"""
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    def fit_python(x: np.ndarray, y: np.ndarray, x0, y0: np.ndarray) -> tuple:
        '''Takes block of data, previous fitting parameters and calculates next fitting parameters'''

        Dx = x - x0
        Dy = y - y0
        a = Dx @ Dy / Dx.dot(Dx)
        b = y0 - a * x0
        # print(f'{x.shape=}')
        # print(f'{y.shape=}')
        # print(f'{y0.shape=}')
        # print(f'{x0=}')
        # print(f'{Dx.shape=}')
        # print(f'{Dy.shape=}')
        # print(f'{a.shape=}')
        # print(f'{b.shape=}')
        return (np.outer(Dx, a) - Dy, (a,  b))
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def fit_numba(x: np.ndarray, y: np.ndarray, x0, y0: np.ndarray) -> tuple:
        '''Takes block of data, previous fitting parameters and calculates next fitting parameters'''

        Dx = x - x0
        Dy = y - y0
        a = Dx.T @ Dy / Dx.T.dot(Dx)
        b = y0 - a * x0
        # print(f'{x.shape=}')
        # print(f'{y.shape=}')
        # print(f'{y0.shape=}')
        # print(f'{x0=}')
        # print(f'{Dx.shape=}')
        # print(f'{Dy.shape=}')
        # print(f'{a.shape=}')
        # print(f'{b.shape=}')
        return (a * Dx - Dy, (a,  b))
    #───────────────────────────────────────────────────────────────────
    fit = (fit_python, fit_numba)
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    def y_from_fit(fit: tuple, x: np.ndarray) -> np.ndarray:
        '''Converts the fitting parameters and x to storable y values'''
        # print(f'{fit[0].shape=}')
        # print(f'{fit[1].shape=}')
        # print(f'{x.shape=}')
        return fit[0] * x + fit[1]
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    def _interpolate(x, x1, x2, y1, y2):
        '''Interpolates between two consecutive points of compressed data'''
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1
    
