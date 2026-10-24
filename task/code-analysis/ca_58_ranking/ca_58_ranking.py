"""Ranking of named scores for a small leaderboard."""

from typing import Dict, Iterable, List, Optional, Tuple

METHODS = ("competition", "dense")
PRECISION = 6


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_rows(rows: List[Tuple[str, float]]) -> None:
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise TypeError(f"row {index} must be a (name, score) pair")
        name, score = row
        if not isinstance(name, str) or name == "":
            raise ValueError(f"row {index} needs a non-empty name")
        if not _is_number(score):
            raise TypeError(f"row {index} score must be a number")
        if score != score:
            raise ValueError(f"row {index} score must not be NaN")
        if name in seen:
            raise ValueError(f"duplicate name {name!r}")
        seen.add(name)


def _check_options(method: str, top: Optional[int]) -> None:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if top is not None:
        if not isinstance(top, int) or isinstance(top, bool):
            raise TypeError("top must be an integer or None")
        if top < 1:
            raise ValueError("top must be at least 1")


def _sorted_rows(rows: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    rounded = [(name, round(float(score), PRECISION - 1)) for name, score in rows]
    return sorted(rounded, key=lambda row: (-row[1], row[0]))


def _assign_ranks(ordered: List[Tuple[str, float]], method: str) -> List[int]:
    ranks: List[int] = []
    for position, (_, score) in enumerate(ordered):
        if position > 0 and score == ordered[position - 1][1]:
            ranks.append(ranks[-1])
        elif method == "competition":
            ranks.append(position + 1)
        else:
            ranks.append(ranks[-1] + 1 if ranks else 1)
    return ranks


def _cut_to_top(
    ordered: List[Tuple[str, float]], ranks: List[int], top: Optional[int]
) -> List[Dict[str, object]]:
    result = []
    for (name, score), rank in zip(ordered, ranks):
        if top is not None and rank > top:
            continue
        result.append({"name": name, "score": score, "rank": rank})
    return result


def rank_scores(
    rows: Iterable[Tuple[str, float]],
    method: str = "competition",
    top: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Rank (name, score) rows from highest score to lowest.

    Rules:
    - Each row is a list or tuple of a non-empty string name and a plain
      number score (booleans and NaN are rejected). Wrong shapes or types
      raise TypeError; bad values and duplicate names raise ValueError.
    - method is "competition" (ties share a rank and the next rank skips
      ahead: 1, 2, 2, 4) or "dense" (ties share a rank and the next rank is
      one higher: 1, 2, 2, 3). Anything else raises ValueError.
    - Scores are rounded to 6 decimal places before they are compared, so
      scores that differ only after the sixth decimal place are tied.
    - Rows are ordered by score descending, then by name ascending, and every
      row reports its rounded score.
    - top is None or a positive integer. When given, every row whose rank is
      at most top is kept, so tied rows at the cut-off all stay.
    - The result is a new list of {"name", "score", "rank"} dicts.
    """
    _check_options(method, top)
    listed = list(rows)
    _check_rows(listed)
    ordered = _sorted_rows(listed)
    ranks = _assign_ranks(ordered, method)
    return _cut_to_top(ordered, ranks, top)
