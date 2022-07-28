from itertools import product
import API
import sys

args = sys.argv[1:]
if not args or 'typing' in args:
    API.typing(shell = True)
if not args or 'unittests' in args:
    API.unittests(verbosity = 2)
if not args or 'benchmark' in args:
    print(f'Benchmarking')
    for time, use_numba in product((1, 10), (False, True)):
        print(f'{use_numba=}, {time=}')
        API.benchmark(use_numba, time)
if not args or 'profile' in args:
    print(f'Profiling')
    for use_numba, repeats, save in zip(2 * (False, True),
                                        (50, 150, 300, 900),
                                        2*(False,) + 2*(True,)):
        print(f'{use_numba=}, {repeats=}, {save=}')
        API.profile(use_numba, repeats, save)
