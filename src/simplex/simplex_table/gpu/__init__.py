from typing import Literal

from .cupy import CupySimplexTable
from .numba import NumbaSimplexTable

BackendGPU = Literal["cupy", "numba"]
