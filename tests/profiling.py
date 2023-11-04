import limesqueezer as ls
import numpy as np

X_DATA, Y_DATA1 = ls.ref.raw_sine_x2(1e4)
Y_DATA2 = np.array((Y_DATA1, Y_DATA1[::-1])).T
ls.G['timed'] = False
# ======================================================================
def no_numba() -> None:
    for _ in range(100):
        ls.compress(X_DATA, Y_DATA2,
                    tolerances = (1e-3, 1e-4, 1), use_numba = False)
# ======================================================================
def with_numba() -> None:
    for _ in range(100):
        ls.compress(X_DATA, Y_DATA2,
                    tolerances = (1e-3, 1e-4, 1), use_numba = True)
