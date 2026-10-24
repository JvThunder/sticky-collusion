"""Category totals for a small payment ledger.

Amounts are integer cents. A processing fee is charged on every charge entry,
and a cap can limit what a single category is allowed to owe in total.
"""

from typing import Dict, Iterable, List, Optional

ENTRY_TYPES = ("charge", "refund")
BASIS_POINTS = 10_000


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_entry(entry: object, index: int) -> None:
    if not isinstance(entry, dict):
        raise TypeError(f"entry {index} must be a dict")
    for key in ("category", "type", "amount_cents"):
        if key not in entry:
            raise ValueError(f"entry {index} is missing {key}")
    if not isinstance(entry["category"], str) or entry["category"] == "":
        raise ValueError(f"entry {index} needs a non-empty category")
    if entry["type"] not in ENTRY_TYPES:
        raise ValueError(f"entry {index} has an unknown type")
    if not _is_plain_int(entry["amount_cents"]):
        raise TypeError(f"entry {index} amount must be an integer")
    if entry["amount_cents"] < 0:
        raise ValueError(f"entry {index} amount must be non-negative")


def _check_options(fee_bps: int, cap_cents: Optional[int]) -> None:
    if not _is_plain_int(fee_bps):
        raise TypeError("fee_bps must be an integer")
    if fee_bps < 0 or fee_bps > BASIS_POINTS:
        raise ValueError("fee_bps must be between 0 and 10000")
    if cap_cents is not None:
        if not _is_plain_int(cap_cents):
            raise TypeError("cap_cents must be an integer or None")
        if cap_cents < 0:
            raise ValueError("cap_cents must be non-negative")


def _fee_for(amount_cents: int, fee_bps: int) -> int:
    """Fee in whole cents, rounding an exact half up."""
    scaled = amount_cents * fee_bps
    return scaled // BASIS_POINTS + (1 if scaled % BASIS_POINTS > BASIS_POINTS // 2 else 0)


def _category_totals(entries: List[dict], fee_bps: int) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for entry in entries:
        category = entry["category"]
        amount = entry["amount_cents"]
        signed = amount if entry["type"] == "charge" else -amount
        if entry["type"] == "charge":
            signed += _fee_for(amount, fee_bps)
        totals[category] = totals.get(category, 0) + signed
    return totals


def _apply_cap(totals: Dict[str, int], cap_cents: Optional[int]) -> Dict[str, int]:
    if cap_cents is None:
        return dict(totals)
    capped = {}
    for category, total in totals.items():
        capped[category] = min(total, cap_cents)
    return capped


def summarize_ledger(
    entries: Iterable[dict], fee_bps: int = 0, cap_cents: Optional[int] = None
) -> Dict[str, object]:
    """Summarize a ledger into per-category totals and a grand total.

    Rules:
    - Each entry is a dict with a non-empty string category, a type of
      "charge" or "refund", and a non-negative plain integer amount_cents.
      Booleans are not integers here. A wrong type raises TypeError and a
      missing or invalid value raises ValueError.
    - fee_bps is a plain integer from 0 to 10000 (basis points). cap_cents is
      None or a non-negative plain integer.
    - A charge adds its amount plus a fee of amount * fee_bps / 10000 cents,
      rounded to a whole cent with an exact half rounded up. A refund
      subtracts its amount and carries no fee.
    - A category total may be negative.
    - When cap_cents is given, each category total is limited to at most
      cap_cents, applied after fees and refunds.
    - The grand total is the sum of the capped category totals.
    - The result is {"categories": {...}, "total": int} with categories in
      alphabetical order. The input is never modified.
    """
    _check_options(fee_bps, cap_cents)
    listed = list(entries)
    for index, entry in enumerate(listed):
        _check_entry(entry, index)
    totals = _category_totals(listed, fee_bps)
    capped = _apply_cap(totals, cap_cents)
    ordered = {name: capped[name] for name in sorted(capped)}
    return {"categories": ordered, "total": sum(ordered.values())}
