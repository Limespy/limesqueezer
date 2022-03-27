'''Unittests for limesqueezer'''
#%%═════════════════════════════════════════════════════════════════════
# IMPORT
import unittest
import numpy as np
import limesqueezer as ls
import reference as ref

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
        self.assertTrue(isinstance(ls.sqrtrange(0, 1), np.ndarray))
        reltol = 5e-2
        for i in [1, 5 , 100, 1000, 10000]:
            ins = ls.sqrtrange(0, i)
            arr = np.arange(i + 1)
            arr[ins]
            self.assertLess((len(ins) / (round((i**0.5)) + 1) - 1), reltol)
            self.assertEqual(ins[0], 0)
            self.assertEqual(ins[-1], i)
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
        for limit in np.logspace(0,3, num = 20).astype(int):
            for x in np.linspace(0,limit, num = 20).astype(int):
                x0, fit0 = ls.droot(f2zero_100, f2zero_100(0)[0], x, limit)
                self.assertLessEqual(x0, limit)
                self.assertLess(f2zero_100(x0)[0], 0)
                if x0 < limit:
                    self.assertGreater(f2zero_100(x0+1)[0], 0)
                    self.assertEqual(x0, 100)
                self.assertTrue(fit0)
    #───────────────────────────────────────────────────────────────────
    def test_1_4_nlines(self):
        x, y = ref.raw_sine(1e3 + 2)
        for end in np.logspace(1,3, 10).astype(int) + 1:
            print(f'{end=}')
            print(ls.n_lines(x[1:end], y[1:end], x[0], y[0], 1e-2))
    #───────────────────────────────────────────────────────────────────
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
        
#%%═════════════════════════════════════════════════════════════════════
# RUNNING TESTS
def main():
    # Run the tests with increased output verbosity.
    # You can change the verbosity to for example 1 and see what happens.
    unittest.main(verbosity = 2)
if __name__ == "__main__":
    main()
