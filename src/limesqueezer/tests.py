'''Unittests for limesqueezer'''
#%%═════════════════════════════════════════════════════════════════════
# IMPORT
import unittest
import numpy as np

import limesqueezer as ls
print(ls.API._sqrtrange)
#%%═════════════════════════════════════════════════════════════════════
# AUXILIARIES
def f2zero_100(n: int) -> float:
    '''returns < 0 for values 0 to 100 and >0 for values > 100'''
    if round(n) != n: raise ValueError('Not whole number')
    if n < 0: raise ValueError('Input must be >= 0')
    return np.sqrt(n) - 10.01, True
#%%═════════════════════════════════════════════════════════════════════
# TEST CASES
class Test(unittest.TestCase):

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
    def test_1_1_sqrtfill(self):
        self.assertTrue(isinstance(ls.API._sqrtrange[0](0, 1), np.ndarray))
        reltol = 5e-2
        for i in [1, 5 , 100, 1000, 10000]:
            ins_py = ls.API._sqrtrange[0](0, i)
            ins_numba = ls.API._sqrtrange[0](0, i)
            self.assertTrue(np.all(ins_py == ins_numba))
            arr = np.arange(i + 1)
            arr[ins_numba]
            self.assertLess((len(ins_py) / (round((i**0.5)) + 1) - 1), reltol)
            self.assertEqual(ins_py[0], 0)
            self.assertEqual(ins_py[-1], i)
    #───────────────────────────────────────────────────────────────────
    def test_1_2_interval(self):
        x1, x2= 50, 150
        x0, fit0 = ls.interval(f2zero_100,
                               x1, f2zero_100(x1)[0],
                               x2, f2zero_100(x2)[0], False)
        self.assertLess(f2zero_100(x0)[0], 0)
        self.assertGreater(f2zero_100(x0+1)[0], 0)
        self.assertEqual(x0, 100)
        self.assertTrue(fit0)
    #───────────────────────────────────────────────────────────────────
    def test_1_3_droot(self):
        for limit in np.logspace(0, 3, num = 20).astype(int):
            for x in np.linspace(0, limit, num = 20).astype(int):
                x0, fit0 = ls.droot(f2zero_100, f2zero_100(0)[0], x, limit)
                self.assertLessEqual(x0, limit)
                self.assertLess(f2zero_100(x0)[0], 0)
                if x0 < limit:
                    self.assertGreater(f2zero_100(x0+1)[0], 0)
                    self.assertEqual(x0, 100)
                self.assertTrue(fit0)
    
    def test_1_4_nlines(self):
        x, y = ls.ref.raw_sine(1e3 + 2)
        for end in np.logspace(1,3, 10).astype(int) + 1:
            print(f'{end=}')
            print(ls.n_lines(x[1:end], y[1:end], x[0], y[0], 1e-2))
    #═══════════════════════════════════════════════════════════════════
    # Block Compresion
    # def test_2_1_compress(self):
        # import math
        # tol = 1e-3
        # x_data = np.linspace(0,6,int(1e3))
        # y_data = np.array(np.sin(x_data*2*math.pi))
        # xc0, yc0 = ls.compress(x_data, y_data, tol = tol)
        # for e in range(4,9):
        #     x_data = np.linspace(0,6,int(10 ** e))
        #     y_data = np.array(np.sin(x_data*2*math.pi))
        #     xc, yc = ls.compress(x_data, y_data, tol = tol)
        #     self.assertEqual(xc0, xc)
        #     self.assertEqual(yc0, yc)
    #═══════════════════════════════════════════════════════════════════
    # Stream Compression
    def test_3_1_stream_1y(self):
        '''Stream compression runs and outputs correctly'''
        tol = 1e-3
        xdata, ydata = ls.ref.raw_sine_x2(1e4)
        #───────────────────────────────────────────────────────────────
        with ls.Stream(xdata[0], ydata[0], tol = tol) as record:
            self.assertTrue(isinstance(record, ls.API._StreamRecord))
            self.assertEqual(record.state, 'open')
            for x, y in zip(xdata[1:], ydata[1:]):
                record(x, y)
        #───────────────────────────────────────────────────────────────
        self.assertEqual(record.state, 'closed')

        self.assertEqual(xdata[0], record.x[0])
        self.assertEqual(xdata[-1], record.x[-1])
        self.assertEqual(ydata[0], record.y[0])
        self.assertEqual(ydata[-1], record.y[-1])
    #───────────────────────────────────────────────────────────────────
    # def test_3_2_stream_2d(self):
    #     '''Stream compression runs and outputs correctly'''
    #     tol = 1e-3
    #     xdata, ydata = ls.ref.raw_sine_x2(1e4)
    #     #───────────────────────────────────────────────────────────────
    #     with ls.Stream(xdata[0], ydata[0], tol = tol) as record:
    #         self.assertTrue(isinstance(record, ls._StreamContainer))
    #         self.assertEqual(record.state, 'open')
    #         for x, y in zip(xdata[1:], ydata[1:]):
    #             record(x, y)
    #     #───────────────────────────────────────────────────────────────
    #     self.assertTrue(isinstance(record, ls.Compressed))
    #     self.assertEqual(record.state, 'closed')

    #     self.assertEqual(xdata[0], record.x[0])
    #     self.assertEqual(xdata[-1], record.x[-1])
    #     self.assertEqual(ydata[0], record.y[0])
    #     self.assertEqual(ydata[-1], record.y[-1])
    #═══════════════════════════════════════════════════════════════════
    # Stream Compression
    def test_4_3_block_vs_stream_1y(self):
        '''Block and stream compressions must give equal compressed output
        for 1 y variable'''

        tol = 1e-2
        xdata, ydata = ls.ref.raw_sine_x2(1e4)
        xc_block, yc_block = ls.compress(xdata, ydata, tol = tol,
                                initial_step = 100, errorfunction = 'maxmaxabs')
        #───────────────────────────────────────────────────────────────
        with ls.Stream(xdata[0], ydata[0], tol = tol) as record:
            for x, y in zip(xdata[1:], ydata[1:]):
                record(x, y)
        #───────────────────────────────────────────────────────────────
        self.assertTrue(np.all(xc_block == record.x))
        self.assertTrue(np.all(yc_block == record.y))
    #═══════════════════════════════════════════════════════════════════
    # Decompression
    def test_5_2_module_call(self):
        '''Block and stream compressions must give equal compressed output
        for 1 y variable'''
        function = ls(*ls.ref.raw_sine_x2(1e4), tol = 1e-2,
                      errorfunction = 'maxmaxabs')


        
#%%═════════════════════════════════════════════════════════════════════
# RUNNING TESTS
def main():
    # Run the tests with increased output verbosity.
    # You can change the verbosity to for example 1 and see what happens.
    unittest.main(verbosity = 2)
if __name__ == "__main__":
    main()
