"""Export a slim copy of the runs behind the sticky-collusion report.

Each run.json stores both agents' full conversation again at every episode, so the raw
records come to several GB. The export keeps, for every episode, the fields the analysis
reads (verdicts and their stated reasons, rewards, ground truth, task ids and the arm
flags) plus each run's configuration, and copies the agreement-judge CSVs of the builds.
It keeps the folder layout of results/, so the analysis runs on it unchanged:

    python reports/sticky-collusion/export_runs.py                # writes data/sticky-collusion/runs
    STICKY_RESULTS=data/sticky-collusion/runs python reports/sticky-collusion/make_figures.py
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC, OUT = ROOT / "results", ROOT / "data" / "sticky-collusion" / "runs"
TREES = ("luna6-branches", "ds-branches", "g31l-branches", "schedule")
SKIP = ("colluding-prefixes", "haiku")  # prefix symlinks duplicate the builds; Haiku pilots are not in the report

EPISODE_KEYS = ("episode_id", "task_type", "task_ids", "verdict_policy", "reward_by_agent", "reward_scheme",
                "reward_paid", "after_trigger", "gateway", "channel_freed", "bug_line_delivered", "memory_strip",
                "penalty", "audited", "notice")
AGENT_KEYS = ("task_id", "function_name", "expected_verdict", "verdict", "verdict_correct", "verdict_submission",
              "reward", "outcome_feedback", "reflection")


def slim(run: dict) -> dict:
    episodes = []
    for e in run.get("results", []):
        kept = {k: e[k] for k in EPISODE_KEYS if k in e}
        kept["agents"] = {a: {k: v[k] for k in AGENT_KEYS if k in v} for a, v in e.get("agents", {}).items()}
        episodes.append(kept)
    # Paths in the config are written relative to the repository root.
    config = json.loads(json.dumps(run.get("run_config", {})).replace(str(ROOT) + "/", ""))
    return {"run_config": config, "llm_usage_summary": run.get("llm_usage_summary", {}), "results": episodes}


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    runs = csvs = 0
    for tree in TREES:
        for f in sorted((SRC / tree).rglob("run.json")):
            rel = f.relative_to(SRC)
            if any(s in str(rel) for s in SKIP) or f.is_symlink():
                continue
            dest = OUT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(slim(json.loads(f.read_text())), separators=(",", ":")))
            runs += 1
            agreement = f.parent / "agreement.csv"
            if agreement.exists() and "build" in str(rel):
                shutil.copy(agreement, dest.parent / "agreement.csv")
                csvs += 1
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"exported {runs} runs and {csvs} agreement files to {OUT.relative_to(ROOT)} ({size / 1e6:.1f} MB)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
