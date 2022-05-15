import numpy as np
import numba
#%%═════════════════════════════════════════════════════════════════════
# BUILTIN COMPRESSION MODELS
class Poly10:
    """Builtin group of functions for doing the compression"""
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    def _fit_python(x: np.ndarray, y: np.ndarray, x0:float, y0: np.ndarray) -> tuple:
        '''Takes block of data, previous fitting parameters and calculates next fitting parameters
        Returns:
        - residuals
        - next y0 according to the fit'''

        Dx = x - x0
        Dy = y - y0
        X = np.outer(Dx, Dx @ Dy / Dx.dot(Dx))
        return X - Dy, X[-1] + y0
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    @numba.jit(nopython=True, cache=True, fastmath = True)
    def _fit_numba(x: np.ndarray, y: np.ndarray, x0: float, y0: np.ndarray) -> tuple:
        '''Takes block of data, previous fitting parameters and calculates next fitting parameters
        - residuals
        - next y0 according to the fit'''

        Dx = x - x0
        Dy = y - y0
        X = np.outer(Dx, Dx @ Dy / Dx.dot(Dx))
        return X - Dy, X[-1] + y0
    #───────────────────────────────────────────────────────────────────
    fit = (_fit_python, _fit_numba)
    #═══════════════════════════════════════════════════════════════════
    @staticmethod
    def _interpolate(x, x1: float, x2: float, y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
        '''Interpolates between two consecutive points of compressed data'''
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1
#%%═════════════════════════════════════════════════════════════════════
class Poly1100:
    """Builtin group of functions for doing the compression"""
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    def _fit_python(x: np.ndarray, y: np.ndarray, x0: float, y0: tuple) -> tuple:
        '''Takes block of data, previous fitting parameters and calculates next fitting parameters'''
        y = y.T
        Dx = x - x0
        Dx2 = Dx * Dx
        Dx3 = Dx2 * Dx
        Y = (y - y0[0] - Dx * y0[1]).T

        xdiff = np.diff(x)
        print(f'{y.shape=}')
        print(f'{xdiff.shape=}')
        print(f'{np.diff(y, axis = 0).shape=}')
        dydx = np.diff(y, axis = 1) / np.diff(x)
        print(f'{dydx.shape=}')
        xmid = x[:-1] + xdiff/2
        Dxmid = (xmid - x0) 
        dY = dydx.T - y0[1]
        Dxmid2 = Dxmid * Dxmid
        # print(f'{y.shape=}')
        # print(f'{Y.shape=}')
        # print(f'{Dx.shape=}')
        X = np.vstack((np.concatenate((Dx3, 3 * Dxmid2)), np.concatenate((Dx2, 2 * Dxmid)))).T
        print(f'{Y.shape=}')
        print(f'{dY.shape=}')
        Y = np.concatenate((Y, dY), axis = 0)
        print(f'{X.shape=}')
        print(f'{Y.shape=}')
        p = np.linalg.lstsq(X, Y, rcond = None)[0]

        reconstruct = p[0] * Dx3 + p[1] * Dx2 + y0[1] * Dx + y0[0]
        # print(f'{Dx=}')
        # print(f'{y0=}')
        # print(f'{reconstruct=}')
        # print(f'{y=}')
        res = reconstruct - y
        # print(f'{res=}')
        # print(f'{reconstruct.shape=}')
        # print(f'{y.shape=}')
        # print(f'{res.shape=}')
        # print(f'\t\t\t{p[0], p[1], y0[1], y0[0]}')
        # print(f'\t\t\t{Dx[-1]}')
        new_y0 =  (reconstruct[-1], 
                      3 * p[0] * Dx2[-1] + 2 * p[1] * Dx[-1] + y0[1])
        # print(f'{new_y0=}')
        return (res.T, new_y0)
    #───────────────────────────────────────────────────────────────────
    @staticmethod
    @numba.jit(nopython=True, cache=True, fastmath = True)
    def _fit_numba(x: np.ndarray, y: np.ndarray, x0: float, y0: tuple) -> tuple:
        '''Takes block of data, previous fitting parameters and calculates next fitting parameters'''
        Dx = x - x0
        Dx2 = Dx * Dx
        Dx3 = Dx2 * Dx
        Y = y.T - y0[0] - Dx * y0[1]
        # print(f'{y.shape=}')
        # print(f'{Y.shape=}')
        # print(f'{Dx.shape=}')
        X = np.vstack((Dx3, Dx2)).T
        Y = Y.flatten()
        # print(f'{X.shape=}')
        # print(f'{Y.shape=}')
        p = np.linalg.lstsq(X, Y, rcond = None)[0]

        reconstruct = p[0] * Dx3 + p[1] * Dx2 + y0[1] * Dx + y0[0]
        # print(f'{Dx=}')
        # print(f'{y0=}')
        # print(f'{reconstruct=}')
        # print(f'{y=}')
        res = reconstruct - y.T
        # print(f'{res=}')
        # print(f'{reconstruct.shape=}')
        # print(f'{y.shape=}')
        # print(f'{res.shape=}')
        # print(f'\t\t\t{p[0], p[1], y0[1], y0[0]}')
        # print(f'\t\t\t{Dx[-1]}')
        new_y0 =  (reconstruct[-1], 
                      3 * p[0] * Dx2[-1] + 2 * p[1] * Dx[-1] + y0[1])
        # print(f'{new_y0=}')
        return (res.T, new_y0)
    #───────────────────────────────────────────────────────────────────
    fit = (_fit_python, _fit_numba)
    #═══════════════════════════════════════════════════════════════════
    @staticmethod
    def _interpolate(x, x1, x2, y1, y2):
        '''Interpolates between two consecutive points of compressed data'''
        _Dx = x - x1
        _Dx2 = _Dx * _Dx
        _Dx3 = _Dx2 * _Dx

        Dx = x2 - x1
        Dx2 = Dx * Dx
        Dx3 = Dx2 * Dx
        Dy = y2[0] - y1[0] - y1[1] * Dx
        Ddy = y2[1] - y1[1]

        # print(f'{X=}')
        # print(f'{np.array((Dy, Ddy))=}')

        p2 = - (Ddy / Dx - 3 * Dy / Dx2)
        p3 = (Dy - Dx2 * p2) / Dx3

        return (p3 * _Dx3 + p2 * _Dx2 + y1[1] * _Dx + y1[0], 
                3 * p3 * _Dx2 + 2 * p2 * _Dx + y1[1])
#═══════════════════════════════════════════════════════════════════════
fitsets = {'Poly10': Poly10,
           'Poly1100': Poly1100}