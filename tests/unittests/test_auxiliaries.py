
import limesqueezer.API as ls

import numpy as np
import pytest

tol = (1e-3, 1e-4, 1)
X_DATA = np.linspace(0, 6, int(1e4))
Y_DATA1 = np.array(np.sin(X_DATA * X_DATA))
Y_DATA2 = np.array((Y_DATA1, Y_DATA1[::-1])).T
#%%═════════════════════════════════════════════════════════════════════
# AUXILIARIES
def f2zero_100(n: int) -> float:
    '''returns < 0 for values 0 to 100 and >0 for values > 100'''
    if round(n) != n: raise ValueError('Not whole number')
    if n < 0: raise ValueError('Input must be >= 0')
    return np.sqrt(n) - 10.01, True
#%%═════════════════════════════════════════════════════════════════════
def compressionaxis(x: np.ndarray, y: np.ndarray) -> int:
    if y.ndim == 1:
        return 0
    if y.shape[0] == len(x):
        return 0
    if y.shape[0] == len(x):
        return 1
def test_f2zero_100():
    assert f2zero_100(100)[0] < 0
    assert f2zero_100(101)[0] > 0
#══════════════════════════════════════════════════════════════════════════════
# _to_ndarray
@pytest.mark.parametrize('array', (X_DATA, list(X_DATA)))
def test_1_1_to_ndarray(array):
    n = len(X_DATA)
    assert np.all(ls.to_ndarray(array) == np.array(array))
    for shape in [(n,), (n,1), (1, n)]:
        assert ls.to_ndarray(array, shape = shape).shape == shape
    for shape in [(-1,), (-1,1), (1, -1)]:
        compare = tuple([n if v == -1 else v for v in shape])
        assert ls.to_ndarray(array, shape = shape).shape == compare
    # #───────────────────────────────────────────────────────────────
    # # 2D input
    # for array in (Y_DATA2, list(Y_DATA2)):
    #     assert ls.to_ndarray(array), np.array(array))
    #     for shape in [(n,), (n,1), (1, n)]:
    #         assert ls.to_ndarray(array, shape = shape).shape, shape)
    #     for shape in [(-1,), (-1,1), (1, -1)]:
    #         compare = tuple([n if v == -1 else n for v in shape])
    #         assert ls.to_ndarray(array, shape = shape).shape, compare)
#══════════════════════════════════════════════════════════════════════════════
# sqrtrange
def test_1_2_sqrtrange():
    '''- sqrtrange works as it should'''
    assert isinstance(ls.sqrtranges[0](1), np.ndarray)
    reltol = 5e-2
    for i in [1, 5 , 100, 1000, 10000]:
        ins_py = ls.sqrtranges[0](i)
        ins_numba = ls.sqrtranges[0](i)
        assert np.all(ins_py == ins_numba)
        arr = np.arange(i + 1)
        arr[ins_numba]
        assert (len(ins_py) / (round((i**0.5)) + 1) - 1) < reltol
        assert ins_py[0] == 0
        assert ins_py[-1] == i
#══════════════════════════════════════════════════════════════════════════════
def test_interval():
    '''- Interval works as it should'''
    x1, x2= 50, 150
    x0, fit0 = ls._intervals[0](f2zero_100,
                            x1, f2zero_100(x1)[0],
                            x2, f2zero_100(x2)[0], False)
    assert f2zero_100(x0)[0] < 0
    assert f2zero_100(x0+1)[0] > 0
    assert fit0
#══════════════════════════════════════════════════════════════════════════════
# def test_1_4_interval():
#     x1, x2= 50, 150
#     x0, fit0 = ls.interval(f2zero_100,
#                            x1, f2zero_100(x1)[0],
#                            x2, f2zero_100(x2)[0], False)
#     assertLess(f2zero_100(x0)[0], 0)
#     assertGreater(f2zero_100(x0+1)[0], 0)
#     assert x0, 100)
#     assert fit0)
#══════════════════════════════════════════════════════════════════════════════
# droot
@pytest.mark.parametrize('limit', (limit for limit in np.logspace(0, 3, num = 20).astype(int)))
def test_1_5_droot(limit):
    '''- Droot works as it should'''
    for x in np.linspace(0, limit, num = 20).astype(int):
        x0, fit0 = ls._droots[0](f2zero_100, f2zero_100(0)[0], x, limit)
        assert x0 <= limit
        assert x0 <= 100
        assert f2zero_100(x0)[0] < 0
        assert fit0
#══════════════════════════════════════════════════════════════════════════════
# nlines
def test_1_6_nlines():
    for end in np.logspace(1,3, 10).astype(int) + 1:
        lines = ls.n_lines(X_DATA[1:end], Y_DATA1[1:end], X_DATA[0], Y_DATA1[0], 1e-2)
        assert lines >= 1