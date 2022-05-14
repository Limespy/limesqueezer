from itertools import product
import unittest
import API
from API import Unittests
import sys

args = sys.argv[1:]

if not args or 'unittest' in args:
    unittest.main(verbosity = 2)
if not args or 'benchmark' in args:
    print(f'Benchmarking')
    for time, use_numba in product((5, 50), (False, True)):
        print(f'\r{use_numba=}, {time=}', end = '')
        API.benchmark(use_numba, time)
if not args or 'profile' in args:
    print(f'Profiling')
    for use_numba, repeats, save in zip(2 * (False, True),
                                        (50, 150, 2000, 5000),
                                        2*(False,) + 2*(True,)):
        print(f'{use_numba=}, {repeats=}, {save=}')
        API.profile(use_numba, repeats, save)