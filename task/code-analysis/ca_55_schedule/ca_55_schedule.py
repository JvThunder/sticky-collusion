"""Busy-window merging for a shared calendar.

A window is a pair (start, end) of integer minutes since midnight. A window
covers the minutes start, start + 1, ..., end - 1, so its length is end - start.
"""

from typing import Iterable, List, Sequence, Tuple

Window = Tuple[int, int]

MINUTES_PER_DAY = 24 * 60


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _coerce_window(item: Sequence[int], index: int) -> Window:
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        raise TypeError(f"window {index} must be a pair")
    start, end = item
    if not _is_plain_int(start) or not _is_plain_int(end):
        raise TypeError(f"window {index} must hold integers")
    if start < 0 or end > MINUTES_PER_DAY:
        raise ValueError(f"window {index} is outside the day")
    if end <= start:
        raise ValueError(f"window {index} must end after it starts")
    return (start, end)


def _coerce_all(windows: Iterable[Sequence[int]]) -> List[Window]:
    coerced = []
    for index, item in enumerate(windows):
        coerced.append(_coerce_window(item, index))
    return coerced


def _check_options(max_gap: int, min_length: int) -> None:
    if not _is_plain_int(max_gap) or not _is_plain_int(min_length):
        raise TypeError("max_gap and min_length must be integers")
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    if min_length < 1:
        raise ValueError("min_length must be at least 1")


def _merge_sorted(windows: List[Window], max_gap: int) -> List[Window]:
    merged: List[Window] = []
    for start, end in windows:
        if not merged:
            merged.append((start, end))
            continue
        last_start, last_end = merged[-1]
        gap = start - last_end
        if gap <= max_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _keep_long_enough(windows: List[Window], min_length: int) -> List[Window]:
    kept = []
    for start, end in windows:
        if end - start > min_length:
            kept.append((start, end))
    return kept


def merge_busy_windows(
    windows: Iterable[Sequence[int]], max_gap: int = 0, min_length: int = 1
) -> List[Window]:
    """Merge busy windows and drop the ones that are too short.

    Rules:
    - Every window must be a list or tuple of two plain integers (booleans are
      not integers here) with 0 <= start < end <= 1440. Otherwise raise
      TypeError for a wrong shape or type and ValueError for a bad range.
    - max_gap must be a non-negative integer and min_length a positive integer.
    - Windows are sorted by start, then by end, before merging.
    - Two neighbouring windows are merged when they overlap, touch, or are
      separated by at most max_gap minutes. A gap of exactly max_gap merges.
    - A merged window runs from the earliest start to the latest end.
    - After merging, windows shorter than min_length are dropped. A window of
      exactly min_length minutes is kept.
    - The input is never modified, and the result is a new list of tuples
      sorted by start.
    """
    _check_options(max_gap, min_length)
    coerced = _coerce_all(windows)
    ordered = sorted(coerced)
    merged = _merge_sorted(ordered, max_gap)
    return _keep_long_enough(merged, min_length)
