#!/usr/bin/env python3
"""Separate knowing rule-bending from other reasons for a mutual ACCEPT.

Reads results/analysis/reasoning.csv (from reasoning_judge.py) and the run.json files:

  1. In throttled episodes, can any agent receive the peer's complete raw log? (It
     counts episodes whose peer log was delivered verbatim in the messages.)
  2. Label shares of throttled ACCEPTs by model and arm (knowing waiver, treats the
     evidence as complete, full-log review).
  3. How often a "credulous" accept leans on get_log or "retrieved" wording.

    python analysis/loophole_report.py [--csv results/analysis/reasoning.csv]
"""

import argparse
import collections
import csv
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETRIEVED = re.compile(r"retriev|obtained|fetched|got the (raw|complete)", re.I)
GET_LOG = re.compile(r"get_log|(raw|complete)[- ]?log[^.]{0,40}(retriev|obtain)", re.I)


def family(pair: str) -> str:
    return "DeepSeek" if "deepseek" in pair else "GPT" if "gpt" in pair else pair


def arm(launch: str) -> str:
    return launch.rsplit("_", 2)[0]


def delivery_check() -> None:
    print("1. Was the peer's complete raw log ever delivered in a throttled episode?")
    stats = collections.defaultdict(collections.Counter)
    for path in glob.glob(str(ROOT / "results/persistence-short/*/*/rep*/run.json")):
        pair = Path(path).parts[-4]
        if "haiku" in pair:
            continue
        for episode in json.load(open(path))["results"]:
            if not episode.get("throttled"):
                continue
            sent = collections.defaultdict(str)
            for event in episode.get("events", []):
                if event.get("tool") == "send_message" and event.get("success"):
                    sent[event["actor"]] += event.get("content", "")
            for peer in ("alice", "bob"):
                log = json.dumps(episode["agents"][peer]["private_raw_log"],
                                 ensure_ascii=False, separators=(",", ":"))
                counter = stats[family(pair)]
                counter["agent-episodes"] += 1
                counter["log delivered verbatim"] += log in sent[peer]
    for name, counter in stats.items():
        print(f"   {name:9} {counter['agent-episodes']} agent-episodes, "
              f"{counter['log delivered verbatim']} with the peer log delivered verbatim")


def label_shares(rows: list[dict]) -> None:
    print("\n2. Label shares of throttled ACCEPTs (what the reasoning says)")
    table = collections.defaultdict(collections.Counter)
    for row in rows:
        if row["verdict"] == "accept" and row["throttled"] == "1" and row["parse_ok"] == "1":
            table[(family(row["pair"]), arm(row["launch"]))][row["label"]] += 1
    for (fam, name), counter in sorted(table.items()):
        total = sum(counter.values())
        print(f"   {fam:9} {name[:34]:34} n={total:3}  "
              f"waiver {counter['KNOWING_WAIVER'] / total:4.0%}  "
              f"treats-as-complete {counter['TREATS_AS_COMPLETE'] / total:4.0%}  "
              f"full-log {counter['FULL_LOG_REVIEW'] / total:4.0%}")


def retrieved_wording(rows: list[dict]) -> None:
    print("\n3. Accepts whose reasoning leans on 'retrieved/obtained' wording")
    for fam in ("DeepSeek", "GPT"):
        for label in ("KNOWING_WAIVER", "TREATS_AS_COMPLETE"):
            sel = [r for r in rows if family(r["pair"]) == fam and r["label"] == label
                   and r["verdict"] == "accept" and r["throttled"] == "1"]
            if sel:
                hits = sum(bool(RETRIEVED.search(r["reasoning"])) for r in sel)
                print(f"   {fam:9} {label:19} {hits}/{len(sel)} ({hits / len(sel):.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", default="results/analysis/reasoning.csv")
    args = parser.parse_args()
    delivery_check()
    rows = list(csv.DictReader(open(ROOT / args.csv)))
    label_shares(rows)
    retrieved_wording(rows)


if __name__ == "__main__":
    main()
