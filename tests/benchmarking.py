# type: ignore
import time

import limesqueezer as ls
import numpy as np
from limedev import BenchmarResults

X_DATA, Y_DATA1 = ls.ref.raw_sine_x2(1e4)
Y_DATA2 = np.array((Y_DATA1, Y_DATA1[::-1])).T

ls.G['timed'] = True

def _run(use_numba: bool) -> float:
    ls.compress(X_DATA, Y_DATA2,
                tolerances = (1e-3, 1e-4, 1),
                use_numba = use_numba,
                errorfunction = 'MaxAbs')
    return ls.G['runtime']
def _loop(duration: float, use_numba: bool):
    end_time = time.time() + duration
    runtime: float = 0.
    n_runs: int = 0
    while time.time() < end_time:
        runtime += _run(use_numba)
        n_runs += 1
    return runtime / n_runs

def main() -> BenchmarResults:
    _loop(1., False)
    _loop(1., True)
    return ls.__version__, {'no_numba': _loop(10, False),
                            'with_numba': _loop(10, True)}
