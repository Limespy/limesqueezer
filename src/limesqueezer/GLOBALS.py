from typing import Any as Any
from typing import Callable as Callable
from numpy.typing import NDArray
import numpy as np
G: dict[str, Any] = {'timed': False,
                     'debug': False,
                     'profiling': False,
                     'runtime': 0}
FloatArray = NDArray[np.float64]
MaybeArray = float | FloatArray
TolerancesInput = float | tuple[MaybeArray, ...]
TolerancesInternal = tuple[FloatArray, FloatArray, FloatArray]