import math
from typing import TypeVar, Sequence

T = TypeVar("T")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def pick_by_fraction(items: Sequence[T], fraction: float) -> T:
    """
    Map fraction in [0.0, 1.0] to an item:
    - 0.0 -> items[0]
    - 1.0 -> items[-1]
    - intermediate values pick bin floor(f * n), clamped to last index
    """
    if not items:
        raise ValueError("items must be non-empty")
    f = _clamp01(fraction)
    idx = min(int(f * len(items)), len(items) - 1)
    return items[idx]


def pick_by_fraction_centered(items: Sequence[T], fraction: float) -> T:
    """
    Map fraction to the nearest item center:
    - scales to [0, n-1] then rounds
    """
    if not items:
        raise ValueError("items must be non-empty")
    f = _clamp01(fraction)
    idx = int(round(f * (len(items) - 1)))
    return items[idx]


def interpolate_numeric_list(values: Sequence[float], fraction: float) -> float:
    """
    Linearly interpolate inside a numeric list.
    - fraction=0.0 -> values[0]
    - fraction=1.0 -> values[-1]
    - smoothly blends between neighbors for intermediate fractions
    """
    if not values:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        return float(values[0])
    f = _clamp01(fraction) * (len(values) - 1)
    i = int(math.floor(f))
    t = f - i
    if i >= len(values) - 1:
        return float(values[-1])
    return float(values[i]) * (1 - t) + float(values[i + 1]) * t
