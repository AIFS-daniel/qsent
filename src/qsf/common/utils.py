"""
Shared utility functions.
"""
import math
from typing import Any


def safe(val: Any, decimals: int = 4) -> float | None:
    """Convert numpy/pandas numerics to JSON-safe Python floats."""
    if val is None:
        return None
    f = float(val)
    return None if (math.isnan(f) or math.isinf(f)) else round(f, decimals)
