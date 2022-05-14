'''Unittests for limesqueezer'''
#%%═════════════════════════════════════════════════════════════════════
# IMPORT
import unittest
import numpy as np
import cProfile
import time
import pathlib
import os

import limesqueezer as ls
from limesqueezer.API import to_ndarray

path_testing = pathlib.Path(__file__).parent

tol = 1e-3
x_data, y_data1 = ls.ref.raw_sine_x2(1e4)
y_data2 = np.array((y_data1, y_data1[::-1])).T

#%%═════════════════════════════════════════════════════════════════════
# AUXILIARIES
def f2zero_100(n: int) -> float:
    '''returns < 0 for values 0 to 100 and >0 for values > 100'''
    if round(n) != n: raise ValueError('Not whole number')
    if n < 0: raise ValueError('Input must be >= 0')
    return np.sqrt(n) - 10.01, True
#%%═════════════════════════════════════════════════════════════════════
# TEST CASES
class Unittests(unittest.TestCase):
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
        n = len(x_data)
        for array in (x_data, list(x_data)):
            self.assertTrue(np.all(to_ndarray(array) == np.array(array)))
            for shape in [(n,), (n,1), (1, n)]:
                self.assertEqual(to_ndarray(array, shape = shape).shape, shape)
            for shape in [(-1,), (-1,1), (1, -1)]:
                compare = tuple([n if v == -1 else v for v in shape])
                self.assertEqual(to_ndarray(array, shape = shape).shape, compare)
        # #───────────────────────────────────────────────────────────────
        # # 2D input
        # for array in (y_data2, list(y_data2)):
        #     self.assertEqual(to_ndarray(array), np.array(array))
        #     for shape in [(n,), (n,1), (1, n)]:
        #         self.assertEqual(to_ndarray(array, shape = shape).shape, shape)
        #     for shape in [(-1,), (-1,1), (1, -1)]:
        #         compare = tuple([n if v == -1 else n for v in shape])
        #         self.assertEqual(to_ndarray(array, shape = shape).shape, compare)
    #───────────────────────────────────────────────────────────────────
    def test_1_2_sqrtfill(self):
        self.assertTrue(isinstance(ls.API._sqrtrange[0](1), np.ndarray))
        reltol = 5e-2
        for i in [1, 5 , 100, 1000, 10000]:
            ins_py = ls.API._sqrtrange[0](i)
            ins_numba = ls.API._sqrtrange[0](i)
            self.assertTrue(np.all(ins_py == ins_numba))
            arr = np.arange(i + 1)
            arr[ins_numba]
            self.assertLess((len(ins_py) / (round((i**0.5)) + 1) - 1), reltol)
            self.assertEqual(ins_py[0], 0)
            self.assertEqual(ins_py[-1], i)
    #───────────────────────────────────────────────────────────────────
    def test_1_3_interval(self):
        x1, x2= 50, 150
        x0, fit0 = ls.interval(f2zero_100,
                               x1, f2zero_100(x1)[0],
                               x2, f2zero_100(x2)[0], False)
        self.assertLess(f2zero_100(x0)[0], 0)
        self.assertGreater(f2zero_100(x0+1)[0], 0)
        self.assertEqual(x0, 100)
        self.assertTrue(fit0)
    #───────────────────────────────────────────────────────────────────
    def test_1_4_interval(self):
        x1, x2= 50, 150
        x0, fit0 = ls.interval(f2zero_100,
                               x1, f2zero_100(x1)[0],
                               x2, f2zero_100(x2)[0], False)
        self.assertLess(f2zero_100(x0)[0], 0)
        self.assertGreater(f2zero_100(x0+1)[0], 0)
        self.assertEqual(x0, 100)
        self.assertTrue(fit0)
    #───────────────────────────────────────────────────────────────────
    def test_1_5_droot(self):
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
            lines = ls.n_lines(x_data[1:end], y_data1[1:end], x_data[0], y_data1[0], 1e-2)
            self.assertGreaterEqual(lines, 1)
    #═══════════════════════════════════════════════════════════════════
    # Block Compresion
    def test_2_1_compress_default_y1(self):
        xc, yc = ls.compress(x_data, y_data1, tol = tol)
        self.assertEqual(x_data[0], xc[0])
        self.assertEqual(y_data1[0], yc[0])
        self.assertEqual(x_data[-1], xc[-1])
        self.assertEqual(y_data1[-1], yc[-1])
    #───────────────────────────────────────────────────────────────────
    def test_2_2_compress_default_y1(self):
        xc, yc = ls.compress(x_data, to_ndarray(y_data1, (-1,1)), tol = tol)
        self.assertEqual(x_data[0], xc[0])
        self.assertEqual(y_data1[0], yc[0])
        self.assertEqual(x_data[-1], xc[-1])
        self.assertEqual(y_data1[-1], yc[-1])
    #───────────────────────────────────────────────────────────────────
    def test_2_3_compress_default_y2(self):
        xc, yc = ls.compress(x_data, y_data2, tol = tol)
        self.assertEqual(x_data[0], xc[0])
        self.assertTrue(np.all(y_data2[0] == yc[0]))
        self.assertEqual(x_data[-1], xc[-1])
        self.assertTrue(np.all(y_data2[-1] == yc[-1]))
    #───────────────────────────────────────────────────────────────────
    # def test_2_4_compress_default_y2(self):
    #     print(y_data2.shape)
    #     xc, yc = ls.compress(x_data, y_data2.T, tol = tol)
    #     y_data2.T
    #     print(f'{np.all(np.diff(x_data)>0)=}')
    #     print(f'{np.all(np.diff(xc)>0)=}')
    #     ydc = ls.decompress(xc, yc)(x_data)
    #     # plotters.comparison(x_data[:9000], y_data2[:9000], ydc[:9000])
    #     print(y_data2.shape)
    #     print(y_data2[:2])
    #     print(ydc[:2])
    #     print(y_data2[0] - yc[0])
    #     print(y_data2[-2:])
    #     print(ydc[-2:])
    #     print(y_data2[-1] - yc[-1])
    #     self.assertEqual(x_data[0], xc[0])
    #     self.assertTrue(np.allclose(y_data2[0], yc[0]))
    #     self.assertEqual(x_data[-1], xc[-1])
    #     self.assertTrue(np.all(y_data2[-1] == yc[-1]))
    #═══════════════════════════════════════════════════════════════════
    # Stream Compression
    def test_3_1_stream_1y(self):
        '''Stream compression runs and outputs correctly'''
        #───────────────────────────────────────────────────────────────
        with ls.Stream(x_data[0], y_data1[0], tol = tol) as record:
            self.assertTrue(isinstance(record, ls.API._StreamRecord))
            self.assertEqual(record.state, 'open')
            for x, y in zip(x_data[1:], y_data1[1:]):
                record(x, y)
        #───────────────────────────────────────────────────────────────
        self.assertEqual(record.state, 'closed')

        self.assertEqual(x_data[0], record.x[0])
        self.assertEqual(y_data1[0], record.y[0])
        self.assertEqual(x_data[-1], record.x[-1])
        self.assertEqual(y_data1[-1], record.y[-1])
    #───────────────────────────────────────────────────────────────────
    # def test_3_2_stream_2d(self):
    #     '''Stream compression runs and outputs correctly'''
    #     tol = 1e-3
    #     x_data, y_data1 = ls.ref.raw_sine_x2(1e4)
    #     #───────────────────────────────────────────────────────────────
    #     with ls.Stream(x_data[0], y_data1[0], tol = tol) as record:
    #         self.assertTrue(isinstance(record, ls._StreamContainer))
    #         self.assertEqual(record.state, 'open')
    #         for x, y in zip(x_data[1:], y_data1[1:]):
    #             record(x, y)
    #     #───────────────────────────────────────────────────────────────
    #     self.assertTrue(isinstance(record, ls.Compressed))
    #     self.assertEqual(record.state, 'closed')

    #     self.assertEqual(x_data[0], record.x_data[0])
    #     self.assertEqual(x_data[-1], record.x_data[-1])
    #     self.assertEqual(y_data1[0], record.y_data1[0])
    #     self.assertEqual(y_data1[-1], record.y_data1[-1])
    #═══════════════════════════════════════════════════════════════════
    # Stream Compression
    def test_4_3_block_vs_stream_1y(self):
        '''Block and stream compressions must give equal compressed output
        for 1 y variable'''

        tol = 1e-2
        x_data, y_data1 = ls.ref.raw_sine_x2(1e4)
        xc_block, yc_block = ls.compress(x_data, y_data1, tol = tol,
                                initial_step = 100, errorfunction = 'maxmaxabs')
        #───────────────────────────────────────────────────────────────
        with ls.Stream(x_data[0], y_data1[0], tol = tol) as record:
            for x, y in zip(x_data[1:], y_data1[1:]):
                record(x, y)
        #───────────────────────────────────────────────────────────────
        self.assertTrue(np.all(xc_block == record.x))
        self.assertTrue(np.all(yc_block == record.y))
    #═══════════════════════════════════════════════════════════════════
    # Decompression
    def test_5_1_decompress_mock(self):
        '''Runs decompression on and compares to original.'''
        self.assertTrue(np.allclose(ls.decompress(x_data, y_data1)(x_data),
                                    y_data1, atol = 1e-14))
    #═══════════════════════════════════════════════════════════════════
    # Compression decompression
    def test_6_2_module_call(self):
        '''Block and stream compressions must give equal compressed output
        for 1 y variable'''
        x_data, y_data1 = ls.ref.raw_sine_x2(1e4)
        tol = 1e-2
        xc, yc = ls.compress(x_data, y_data1, tol = tol, errorfunction = 'maxmaxabs')
        function = ls.decompress(xc, yc)
        function_call = ls(x_data, y_data1, tol = tol, errorfunction = 'maxmaxabs')
#═══════════════════════════════════════════════════════════════════════
def benchmark(use_numba: bool, timerange: float):
    endtime = time.time() + timerange
    n = 0
    n2 = 50
    ls.G['timed'] = True
    runtime = []
    while time.time() < endtime:
        print(f'Benchmarking, loopset {n}')
        for _ in range(n2):
            ls.compress(x_data, y_data2, tol = 1e-3, use_numba = use_numba)
            runtime.append(ls.G['runtime'])
        n += 1
    runtime = np.array(runtime)
    print(f'mean runtime {"with" if use_numba else "without"} numba was {sum(runtime) / (n * n2)*1e3:.1f} ms') # mean runtime
    return runtime, np.cumsum(runtime)
#═══════════════════════════════════════════════════════════════════════
def profile(use_numba, n_runs, is_save):
    ls.G['timed'] = False
    profilecode = f'for _ in range({n_runs}): API.ls.compress(API.x_data, API.y_data2, tol = 1e-3, use_numba = {use_numba})'
    path_out = path_testing / 'profiling' / f'{"with" if use_numba else "no"}_numba.pstats'
    cProfile.run(profilecode, path_out)
    if is_save:
        os.system(f'gprof2dot -f pstats {path_out} | dot -Tpdf -o {path_out.with_suffix(".pdf")}')
