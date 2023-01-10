'''Unittests for limesqueezer'''
#%%═════════════════════════════════════════════════════════════════════
# IMPORT

import gprof2dot
import numpy as np
import pydot

import cProfile
import os
import pathlib
import time


PATH_TESTS = pathlib.Path(__file__).parent
PATH_UNITTESTS = PATH_TESTS / 'unittests'
PATH_REPO = PATH_TESTS.parent
# First item in src should be the package
PATH_SRC = next((PATH_REPO / 'src').glob('*'))

#═══════════════════════════════════════════════════════════════════════
def unittests(verbosity: int = 2) -> None:
    import pytest
    CWD = pathlib.Path.cwd()
    os.chdir(str(PATH_UNITTESTS))
    output = pytest.main([])
    os.chdir(str(CWD))
    return output
#%%═════════════════════════════════════════════════════════════════════
def benchmark(use_numba: bool, timerange: float) -> tuple[float, np.ndarray]:
    import limesqueezer as ls
    X_DATA, Y_DATA1 = ls.ref.raw_sine_x2(1e4)
    Y_DATA2 = np.array((Y_DATA1, Y_DATA1[::-1])).T

    endtime = time.time() + timerange
    n = 0
    n2 = 50
    ls.G['timed'] = True
    runtime = []
    while time.time() < endtime:
        print(f'\rBenchmarking, loopset {n}', end = '')
        for _ in range(n2):
            ls.compress(X_DATA, Y_DATA2, tolerances = (1e-3, 1e-4, 1), use_numba = use_numba, errorfunction = 'MaxAbs')
            runtime.append(ls.G['runtime'])
        n += 1
    runtime = np.array(runtime)
    print(f'\nMean runtime {"with" if use_numba else "without"} numba was {sum(runtime) / (n * n2)*1e3:.1f} ms') # mean runtime
    return runtime, np.cumsum(runtime)
#═══════════════════════════════════════════════════════════════════════
PATH_PROFILING = PATH_TESTS / 'profiling'
fnames = [f'{info}_numba' for info in ('no', 'with')]
path_pstats = [PATH_PROFILING / f'{fname}.pstats' for fname in fnames]
path_dot = [PATH_PROFILING / f'{fname}.dot' for fname in fnames]
path_pdf = [str(PATH_PROFILING / f'{fname}.pdf') for fname in fnames]

def profile(use_numba: bool, n_runs: int, is_save: bool):
    import limesqueezer as ls
    X_DATA, Y_DATA1 = ls.ref.raw_sine_x2(1e4)
    Y_DATA2 = np.array((Y_DATA1, Y_DATA1[::-1])).T
    ls.G['timed'] = False

    with cProfile.Profile() as pr:
        for _ in range(n_runs):
            ls.compress(X_DATA, Y_DATA2,
                        tolerances = (1e-3, 1e-4, 1),
                        use_numba = use_numba)
        pr.dump_stats(path_pstats[use_numba])
    if is_save:
        gprof2dot.main(['-f', 'pstats', str(path_pstats[use_numba]),
                        '-o', path_dot[use_numba]])
        output = os.system(f'dot -Tpdf {path_dot[use_numba]} -o {path_pdf[use_numba]}')
        if output:
            raise RuntimeError('Conversion to PDF failed, maybe graphviz dot program is not installed. http://www.graphviz.org/download/')

        path_dot[use_numba].unlink()
    path_pstats[use_numba].unlink()
#═══════════════════════════════════════════════════════════════════════
def typing(shell: bool = False) -> None | tuple[str, str, int]:
    args = [str(PATH_SRC), '--config-file', str(PATH_TESTS / "mypy.ini")]
    if shell:
        os.system(f'mypy {" ".join(args)}')
    else:
        from mypy import api as mypy
        return mypy.run(args)