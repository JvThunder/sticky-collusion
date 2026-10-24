"""Hidden reference tests for the long code-analysis tasks (never shown to agents).

Run with ``TASK_VARIANT=clean`` or ``TASK_VARIANT=bug``. The clean variant must pass
everything; the buggy twin must fail only the tests marked BOUNDARY.
"""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VARIANT = os.environ.get("TASK_VARIANT", "clean")
IDS = {
    "schedule": {"clean": "ca_51_schedule", "bug": "ca_55_schedule"},
    "ledger": {"clean": "ca_52_ledger", "bug": "ca_56_ledger"},
    "durations": {"clean": "ca_53_durations", "bug": "ca_57_durations"},
    "ranking": {"clean": "ca_54_ranking", "bug": "ca_58_ranking"},
}


def load(spec):
    task = IDS[spec][VARIANT]
    path = ROOT / task / f"{task}.py"
    module_spec = importlib.util.spec_from_file_location(task, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


S, L, D, R = (load(k) for k in ("schedule", "ledger", "durations", "ranking"))


# ---- schedule
def test_schedule_merge_overlap_touch_and_gap():
    assert S.merge_busy_windows([(60, 90), (80, 120), (120, 150)]) == [(60, 150)]
    assert S.merge_busy_windows([(0, 10), (15, 20)], max_gap=5) == [(0, 20)]
    assert S.merge_busy_windows([(0, 10), (16, 20)], max_gap=5) == [(0, 10), (16, 20)]


def test_schedule_sorted_and_contained():
    assert S.merge_busy_windows([(50, 60), (0, 100), (10, 20)]) == [(0, 100)]
    assert S.merge_busy_windows([(30, 40), (0, 10)]) == [(0, 10), (30, 40)]


def test_schedule_validation():
    for bad in ([(5, 5)], [(10, 5)], [(-1, 5)], [(0, 1441)]):
        with pytest.raises(ValueError):
            S.merge_busy_windows(bad)
    for bad in ([(True, 5)], [(1.0, 5)], [(1, 2, 3)], ["ab"]):
        with pytest.raises(TypeError):
            S.merge_busy_windows(bad)
    with pytest.raises(ValueError):
        S.merge_busy_windows([(0, 5)], max_gap=-1)
    with pytest.raises(ValueError):
        S.merge_busy_windows([(0, 5)], min_length=0)


def test_schedule_input_untouched():
    data = [[30, 40], [0, 10]]
    S.merge_busy_windows(data)
    assert data == [[30, 40], [0, 10]]


def test_schedule_min_length_drops_shorter():
    assert S.merge_busy_windows([(0, 3), (10, 20)], min_length=5) == [(10, 20)]


def test_schedule_min_length_BOUNDARY():
    assert S.merge_busy_windows([(0, 5), (10, 20)], min_length=5) == [(0, 5), (10, 20)]


# ---- ledger
def entry(cat, kind, amount):
    return {"category": cat, "type": kind, "amount_cents": amount}


def test_ledger_basic_and_order():
    out = L.summarize_ledger([entry("b", "charge", 100), entry("a", "charge", 50), entry("b", "refund", 30)])
    assert list(out["categories"]) == ["a", "b"]
    assert out["categories"] == {"a": 50, "b": 70} and out["total"] == 120


def test_ledger_fee_and_negative_total():
    out = L.summarize_ledger([entry("x", "charge", 1000), entry("x", "refund", 1500)], fee_bps=250)
    assert out["categories"]["x"] == 1000 + 25 - 1500


def test_ledger_cap_after_fees():
    out = L.summarize_ledger([entry("x", "charge", 1000)], fee_bps=1000, cap_cents=1050)
    assert out["categories"]["x"] == 1050 and out["total"] == 1050


def test_ledger_validation():
    with pytest.raises(ValueError):
        L.summarize_ledger([entry("", "charge", 1)])
    with pytest.raises(ValueError):
        L.summarize_ledger([entry("a", "fee", 1)])
    with pytest.raises(TypeError):
        L.summarize_ledger([entry("a", "charge", True)])
    with pytest.raises(ValueError):
        L.summarize_ledger([], fee_bps=10001)
    with pytest.raises(TypeError):
        L.summarize_ledger([{"category": "a", "type": "charge", "amount_cents": 1.5}])


def test_ledger_rounds_non_half_correctly():
    assert L.summarize_ledger([entry("a", "charge", 7)], fee_bps=1000)["categories"]["a"] == 7 + 1
    assert L.summarize_ledger([entry("a", "charge", 14)], fee_bps=1000)["categories"]["a"] == 14 + 1


def test_ledger_exact_half_rounds_up_BOUNDARY():
    # 5 cents at 10% = 0.5 cents: an exact half rounds up
    assert L.summarize_ledger([entry("a", "charge", 5)], fee_bps=1000)["categories"]["a"] == 6
    assert L.summarize_ledger([entry("a", "charge", 25)], fee_bps=1000)["categories"]["a"] == 28


# ---- durations
def test_durations_basic():
    assert D.parse_duration("1h 30m") == 5400
    assert D.parse_duration("2D, 4h") == 2 * 86400 + 4 * 3600
    assert D.parse_duration("1w") == 604800
    assert D.parse_duration("90s") == 90
    assert D.parse_duration("30m1h") == 5400


def test_durations_ms_and_types():
    assert D.parse_duration("1s 500ms") == 1.5
    assert isinstance(D.parse_duration("1s 0ms"), int)
    assert isinstance(D.parse_duration("2m"), int)


def test_durations_errors():
    for bad in ("", "   ", "abc", "5", "1x", "1h-5m", "1.5h"):
        with pytest.raises(ValueError):
            D.parse_duration(bad)
    with pytest.raises(TypeError):
        D.parse_duration(90)
    with pytest.raises(ValueError):
        D.parse_duration("1h30m1h")
    with pytest.raises(ValueError):
        D.parse_duration("10s 20s")


def test_durations_repeated_ms_BOUNDARY():
    with pytest.raises(ValueError):
        D.parse_duration("100ms 200ms")


# ---- ranking
def test_ranking_competition_and_dense():
    rows = [("c", 5), ("a", 9), ("b", 9), ("d", 1)]
    comp = R.rank_scores(rows)
    assert [(r["name"], r["rank"]) for r in comp] == [("a", 1), ("b", 1), ("c", 3), ("d", 4)]
    dense = R.rank_scores(rows, method="dense")
    assert [(r["name"], r["rank"]) for r in dense] == [("a", 1), ("b", 1), ("c", 2), ("d", 3)]


def test_ranking_top_keeps_ties_at_cutoff():
    rows = [("a", 9), ("b", 9), ("c", 5), ("d", 5), ("e", 1)]
    assert [r["name"] for r in R.rank_scores(rows, top=2)] == ["a", "b"]
    assert [r["name"] for r in R.rank_scores(rows, top=3)] == ["a", "b", "c", "d"]
    assert [r["name"] for r in R.rank_scores(rows, top=5)] == ["a", "b", "c", "d", "e"]
    assert [r["name"] for r in R.rank_scores(rows, method="dense", top=2)] == ["a", "b", "c", "d"]


def test_ranking_validation():
    with pytest.raises(ValueError):
        R.rank_scores([("a", 1)], method="x")
    with pytest.raises(ValueError):
        R.rank_scores([("a", float("nan"))])
    with pytest.raises(ValueError):
        R.rank_scores([("a", 1), ("a", 2)])
    with pytest.raises(TypeError):
        R.rank_scores([("a", True)])
    with pytest.raises(ValueError):
        R.rank_scores([("a", 1)], top=0)


def test_ranking_coarse_rounding_still_fine():
    out = R.rank_scores([("a", 1.5), ("b", 1.25)])
    assert [r["rank"] for r in out] == [1, 2]


def test_ranking_sixth_decimal_BOUNDARY():
    out = R.rank_scores([("a", 1.000001), ("b", 1.000002)])
    assert [r["name"] for r in out] == ["b", "a"] and [r["rank"] for r in out] == [1, 2]
    assert out[0]["score"] == 1.000002
