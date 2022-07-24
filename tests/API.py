'''Unittests for limesqueezer'''
#%%═════════════════════════════════════════════════════════════════════
# IMPORT
import limesqueezer as ls
from limesqueezer import to_ndarray

import numpy as np

import unittest

import cProfile
import time
import pathlib
import os

PATH_TESTS = pathlib.Path(__file__).parent
PATH_REPO = PATH_TESTS.parent
# First item in src should be the package
PATH_SRC = next((PATH_REPO / 'src').glob('*'))

tol = (1e-3, 1e-4, 1)
X_DATA, Y_DATA1 = ls.ref.raw_sine_x2(1e4)
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
#%%═════════════════════════════════════════════════════════════════════
# TEST CASES
class Unittests(unittest.TestCase):
    #═══════════════════════════════════════════════════════════════════
    # Auxiliaries
    def assertNpEqual(self, left: np.ndarray, right: np.ndarray, /):
        '''Asserts that two Numpy arrays have same shape and have equal elements'''
        self.assertEqual(left.shape, right.shape)
        self.assertTrue(np.all(left == right), f'{left=} {right=}')
    #───────────────────────────────────────────────────────────────────
    def assertEndpointEqual(self, left, right, /):
        '''Assers that two sequences have both first and last elements equal'''
        self.assertNpEqual(left[0], right[0])
        self.assertNpEqual(left[-1], right[-1])
    #───────────────────────────────────────────────────────────────────
    def aux_compress_default(self, y_data):
        xc, yc = ls.compress(X_DATA, y_data, tolerances = tol, keepshape = True)
        self.assertEqual(y_data.ndim, yc.ndim)
        self.assertEndpointEqual(X_DATA, xc)
        if compressionaxis(X_DATA, y_data):
            self.assertEndpointEqual(y_data.T, yc.T)
    #═══════════════════════════════════════════════════════════════════
    # Metatests on tests themselves
    def test_0_1_f2zero_100(self):
        self.assertTrue(isinstance(f2zero_100(0), tuple))
        self.assertEqual(len(f2zero_100(0)), 2)
        self.assertTrue(isinstance(f2zero_100(0)[0], float))
        self.assertTrue(f2zero_100(0)[1])
        for n in range(101):
            self.assertLess(f2zero_100(n)[0], 0)
        for n in range(101, 201):
            self.assertGreater(f2zero_100(n)[0], 0)
        with self.assertRaises(ValueError):
            f2zero_100(1.000000000001)
        with self.assertRaises(ValueError):
            f2zero_100(-1)
    #═══════════════════════════════════════════════════════════════════
    # Auxiliaries
    def test_1_1_to_ndarray(self):
        n = len(X_DATA)
        for array in (X_DATA, list(X_DATA)):
            self.assertTrue(np.all(to_ndarray(array) == np.array(array)))
            for shape in [(n,), (n,1), (1, n)]:
                self.assertEqual(to_ndarray(array, shape = shape).shape, shape)
            for shape in [(-1,), (-1,1), (1, -1)]:
                compare = tuple([n if v == -1 else v for v in shape])
                self.assertEqual(to_ndarray(array, shape = shape).shape, compare)
        # #───────────────────────────────────────────────────────────────
        # # 2D input
        # for array in (Y_DATA2, list(Y_DATA2)):
        #     self.assertEqual(to_ndarray(array), np.array(array))
        #     for shape in [(n,), (n,1), (1, n)]:
        #         self.assertEqual(to_ndarray(array, shape = shape).shape, shape)
        #     for shape in [(-1,), (-1,1), (1, -1)]:
        #         compare = tuple([n if v == -1 else n for v in shape])
        #         self.assertEqual(to_ndarray(array, shape = shape).shape, compare)
    #───────────────────────────────────────────────────────────────────
    def test_1_2_sqrtrange(self):
        '''- sqrtrange works as it should'''
        self.assertTrue(isinstance(ls.API.sqrtranges[0](1), np.ndarray))
        reltol = 5e-2
        for i in [1, 5 , 100, 1000, 10000]:
            ins_py = ls.API.sqrtranges[0](i)
            ins_numba = ls.API.sqrtranges[0](i)
            self.assertTrue(np.all(ins_py == ins_numba))
            arr = np.arange(i + 1)
            arr[ins_numba]
            self.assertLess((len(ins_py) / (round((i**0.5)) + 1) - 1), reltol)
            self.assertEqual(ins_py[0], 0)
            self.assertEqual(ins_py[-1], i)
    #───────────────────────────────────────────────────────────────────
    def test_1_3_interval(self):
        '''- Interval works as it should'''
        x1, x2= 50, 150
        x0, fit0 = ls.interval(f2zero_100,
                               x1, f2zero_100(x1)[0],
                               x2, f2zero_100(x2)[0], False)
        self.assertLess(f2zero_100(x0)[0], 0)
        self.assertGreater(f2zero_100(x0+1)[0], 0)
        self.assertEqual(x0, 100)
        self.assertTrue(fit0)
    # #───────────────────────────────────────────────────────────────────
    # def test_1_4_interval(self):
    #     x1, x2= 50, 150
    #     x0, fit0 = ls.interval(f2zero_100,
    #                            x1, f2zero_100(x1)[0],
    #                            x2, f2zero_100(x2)[0], False)
    #     self.assertLess(f2zero_100(x0)[0], 0)
    #     self.assertGreater(f2zero_100(x0+1)[0], 0)
    #     self.assertEqual(x0, 100)
    #     self.assertTrue(fit0)
    #───────────────────────────────────────────────────────────────────
    def test_1_5_droot(self):
        '''- Droot works as it should'''
        for limit in np.logspace(0, 3, num = 20).astype(int):
            for x in np.linspace(0, limit, num = 20).astype(int):
                x0, fit0 = ls.droot(f2zero_100, f2zero_100(0)[0], x, limit)
                self.assertLessEqual(x0, limit)
                self.assertLess(f2zero_100(x0)[0], 0)
                if x0 < limit:
                    self.assertGreater(f2zero_100(x0+1)[0], 0)
                    self.assertEqual(x0, 100)
                self.assertTrue(fit0)
    #───────────────────────────────────────────────────────────────────
    def test_1_6_nlines(self):
        for end in np.logspace(1,3, 10).astype(int) + 1:
            lines = ls.n_lines(X_DATA[1:end], Y_DATA1[1:end], X_DATA[0], Y_DATA1[0], 1e-2)
            self.assertGreaterEqual(lines, 1)
    #═══════════════════════════════════════════════════════════════════
    # Block Compresion
    def test_block_1_1_compress_default_y1(self):
        '''- 1D input'''
        self.aux_compress_default(Y_DATA1)
    #───────────────────────────────────────────────────────────────────
    def test_block_1_2_compress_default_y1(self):
        '''- 1D input as column of 2D n x 1 array'''
        self.aux_compress_default(to_ndarray(Y_DATA1, (-1,1)))
    #───────────────────────────────────────────────────────────────────
    def test_block_1_3_compress_default_y2(self):
        '''- 2D input'''
        self.aux_compress_default(Y_DATA2)
    #───────────────────────────────────────────────────────────────────
    def test_block_1_4_compress_default_y2(self):
        '''- 2D input transposed'''
        self.aux_compress_default(Y_DATA2.T)
    #───────────────────────────────────────────────────────────────────
    def test_block_2_1_tolerances_correct_input(self):
        '''- Compression accepts different tolerance inputs
        '''
        ls.compress(X_DATA, Y_DATA1, tolerances = (1e-2, 1e-2, 0))
        ls.compress(X_DATA, Y_DATA1, tolerances = (1e-2, 1e-2))
        ls.compress(X_DATA, Y_DATA1, tolerances = (1e-2))
        ls.compress(X_DATA, Y_DATA1, tolerances = 1e-2)
    #───────────────────────────────────────────────────────────────────
    def test_block_2_1_tolerances_incorrect_input(self):
        '''- Compression rejects incorrect tolerance inputs
        '''
        with self.assertRaises(TypeError):
            ls.compress(X_DATA, Y_DATA1, tolerances = 'hmm')
        with self.assertRaises(ValueError):
            ls.compress(X_DATA, Y_DATA1, tolerances = ())
        with self.assertRaises(ValueError):
            ls.compress(X_DATA, Y_DATA1, tolerances = (1, 2, 3, 4))
    #───────────────────────────────────────────────────────────────────
    def test_block_2_3_tolerances_limits(self):
        '''- Compression works as expected at the edges of the tolerance
        range
        '''
        x_c, y_c = ls.compress(X_DATA, Y_DATA1,
                              tolerances = np.finfo(float).max / 1e2,
                              keepshape = True)
        self.assertEqual((2,), x_c.shape)
        self.assertEqual((2,), y_c.shape)
        x_c, y_c = ls.compress(X_DATA, Y_DATA1,
                               tolerances = np.finfo(float).eps,
                               keepshape = True)
        self.assertEqual(X_DATA.shape, x_c.shape)
        self.assertEqual(Y_DATA1.shape, y_c.shape)
    #───────────────────────────────────────────────────────────────────
    def test_block_3_1_keepshape(self):
        '''- Array noncompressed dimension is kept same'''
        x_c, y_c = ls.compress(X_DATA, Y_DATA2, keepshape = True)
        self.assertEqual(len(X_DATA.shape), len(x_c.shape))
        self.assertEqual(Y_DATA2.shape[1], y_c.shape[1])
    #═══════════════════════════════════════════════════════════════════
    # Stream Compression
    def test_stream_1_1y(self):
        '''- Stream compression runs and outputs correctly'''
        #───────────────────────────────────────────────────────────────
        with ls.Stream(X_DATA[0], Y_DATA1[0], tolerances = tol) as record:
            self.assertTrue(isinstance(record, ls.API._StreamRecord))
            self.assertEqual(record.state, 'open')
            for x, y in zip(X_DATA[1:], Y_DATA1[1:]):
                record(x, y)
        #───────────────────────────────────────────────────────────────
        self.assertEqual(record.state, 'closed')
        self.assertEndpointEqual(X_DATA, record.x)
        self.assertEndpointEqual(to_ndarray(Y_DATA1, (-1, 1)), record.y)
    #═══════════════════════════════════════════════════════════════════
    # Stream Compression
    def test_stream_vs_block_3_1y(self):
        '''- Block and stream compressions must give equal compressed output
        for 1 y variable'''

        X_DATA, Y_DATA1 = ls.ref.raw_sine_x2(1e4)
        xc_block, yc_block = ls.compress(X_DATA, Y_DATA1, tolerances = tol,
                                initial_step = 100, errorfunction = 'maxmaxabs')
        #───────────────────────────────────────────────────────────────
        with ls.Stream(X_DATA[0], Y_DATA1[0], tolerances = tol) as record:
            for x, y in zip(X_DATA[1:], Y_DATA1[1:]):
                record(x, y)
        #───────────────────────────────────────────────────────────────
        self.assertNpEqual(xc_block, record.x)
        self.assertNpEqual(yc_block, record.y)
    #═══════════════════════════════════════════════════════════════════
    # Decompression
    def test_decompress_1_mock(self):
        '''- Runs decompression on and compares to original.'''
        self.assertTrue(np.allclose(ls.decompress(X_DATA, Y_DATA1)(X_DATA),
                                    Y_DATA1, atol = 1e-14))
    #═══════════════════════════════════════════════════════════════════
    # Compression decompression
    def test_module_2_call(self):
        '''- Block and stream compressions give equal compressed output
        for 1 y variable'''
        X_DATA, Y_DATA1 = ls.ref.raw_sine_x2(1e4)
        xc, yc = ls.compress(X_DATA, Y_DATA1,
                             tolerances = tol, errorfunction = 'maxmaxabs')
        function = ls.decompress(xc, yc)
        function_call = ls(X_DATA, Y_DATA1,
                           tolerances = tol, errorfunction = 'maxmaxabs')
#═══════════════════════════════════════════════════════════════════════
def unittests(verbosity: int = 2) -> unittest.TestResult:
    return unittest.TextTestRunner(verbosity = verbosity).run(unittest.makeSuite(Unittests))
#%%═════════════════════════════════════════════════════════════════════
def benchmark(use_numba: bool, timerange: float) -> tuple[float, np.ndarray]:
    endtime = time.time() + timerange
    n = 0
    n2 = 50
    ls.G['timed'] = True
    runtime = []
    while time.time() < endtime:
        print(f'\rBenchmarking, loopset {n}', end = '')
        for _ in range(n2):
            ls.compress(X_DATA, Y_DATA2, tolerances = (1e-3, 1e-4, 1), use_numba = use_numba)
            runtime.append(ls.G['runtime'])
        n += 1
    runtime = np.array(runtime)
    print(f'\nMean runtime {"with" if use_numba else "without"} numba was {sum(runtime) / (n * n2)*1e3:.1f} ms') # mean runtime
    return runtime, np.cumsum(runtime)
#═══════════════════════════════════════════════════════════════════════
def profile(use_numba: bool, n_runs: int, is_save: bool):
    ls.G['timed'] = False
    profilecode = f'for _ in range({n_runs}): API.ls.compress(API.X_DATA, API.Y_DATA2, tolerances = (1e-3, 1e-4, 1), use_numba = {use_numba})'
    path_out = PATH_TESTS / 'profiling' / f'{"with" if use_numba else "no"}_numba.pstats'
    cProfile.run(profilecode, path_out)
    if is_save:
        os.system(f'gprof2dot -f pstats {path_out} | dot -Tpdf -o {path_out.with_suffix(".pdf")}')
#═══════════════════════════════════════════════════════════════════════
def typing(shell: bool = False) -> None | tuple[str, str, int]:
    args = [str(PATH_SRC), '--config-file', str(PATH_TESTS / "mypy.ini")]
    if shell:
        os.system(f'mypy {" ".join(args)}')
    else:
        from mypy import api as mypy
        return mypy.run(args)