"""Parse short duration expressions such as "1h 30m" into seconds."""

import re
from typing import Dict, List, Tuple, Union

Number = Union[int, float]

UNIT_SECONDS: Dict[str, int] = {
    "w": 7 * 24 * 3600,
    "d": 24 * 3600,
    "h": 3600,
    "m": 60,
    "s": 1,
}
MILLISECOND_UNIT = "ms"
TOKEN = re.compile(r"(\d+)(ms|w|d|h|m|s)")


def _normalize(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("duration must be a string")
    compact = re.sub(r"[\s,]+", "", text.strip().lower())
    if compact == "":
        raise ValueError("duration is empty")
    return compact


def _tokenize(compact: str) -> List[Tuple[int, str]]:
    tokens: List[Tuple[int, str]] = []
    position = 0
    while position < len(compact):
        match = TOKEN.match(compact, position)
        if match is None:
            raise ValueError(f"cannot parse duration near {compact[position:]!r}")
        tokens.append((int(match.group(1)), match.group(2)))
        position = match.end()
    return tokens


def _reject_repeats(tokens: List[Tuple[int, str]]) -> None:
    seen = set()
    for _, unit in tokens:
        if unit in seen:
            raise ValueError(f"unit {unit!r} appears more than once")
        if unit != MILLISECOND_UNIT:
            seen.add(unit)


def _total_seconds(tokens: List[Tuple[int, str]]) -> Number:
    whole = 0
    milliseconds = 0
    for amount, unit in tokens:
        if unit == MILLISECOND_UNIT:
            milliseconds += amount
        else:
            whole += amount * UNIT_SECONDS[unit]
    if milliseconds == 0:
        return whole
    return whole + milliseconds / 1000


def parse_duration(text: str) -> Number:
    """Convert a duration expression to seconds.

    Rules:
    - text must be a string, otherwise raise TypeError.
    - Letter case, spaces and commas between parts are ignored, so "1H, 30M"
      is the same as "1h30m". An empty or all-blank string raises ValueError.
    - A duration is one or more parts, each a non-negative integer followed
      by a unit: w (weeks), d (days), h (hours), m (minutes), s (seconds) or
      ms (milliseconds). Anything else raises ValueError.
    - Parts may come in any order, but each unit may appear at most once;
      a repeated unit raises ValueError.
    - The result is the sum in seconds. It is an int when no ms part is
      present, or when the ms part is zero, and a float otherwise.
    """
    compact = _normalize(text)
    tokens = _tokenize(compact)
    _reject_repeats(tokens)
    return _total_seconds(tokens)
