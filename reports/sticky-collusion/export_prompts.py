"""Write the prompts the agents saw, verbatim, for the report's prompts appendix.

The system prompt and one episode's runner messages come from a saved history's run.json, so they are
exactly what the agents received; the texts each arm adds come from the runner's own constants (the audit
message from a recorded penalty run). Output goes to reports/sticky-collusion/prompts/*.txt.

    python reports/sticky-collusion/export_prompts.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments import episode_runner as er  # noqa: E402
from experiments import runner  # noqa: E402

OUT = Path(__file__).resolve().parent / "prompts"
RES = ROOT / "results"
HISTORY = RES / "luna6-branches/colluding-prefixes/gpt-b2-all/rep4/run.json"  # a saved GPT-6-Luna history
OTHER_MODELS = (RES / "ds-branches/colluding-prefixes/ds-build-all/rep1/run.json",
                RES / "g31l-branches/colluding-prefixes/g-build-all/rep1/run.json")


def system_prompts(path):
    return json.load(open(path))["run_config"]["prompt_texts"]["system_prompts"]


def episode_messages(run, agent="alice"):
    """The runner's messages to one agent in the first episode, with the peer's messages as delivered."""
    messages = run["results"][0]["agents"][agent]["messages"]
    parts, tools = [], []
    for m in messages[1:]:
        if m.get("role") == "assistant":  # the agent's own turns are not prompts; name the tools it called
            tools += [c["function"]["name"] for c in m.get("tool_calls") or []]
        elif m.get("role") == "user":
            if tools:
                names = ", ".join(dict.fromkeys(tools))
                parts.append(f"[the agent calls {names}; the tool results are omitted]")
                tools = []
            parts.append(str(m["content"]).strip())
    return "\n\n-----\n\n".join(parts) + "\n"


def penalty_audit_message():
    for path in sorted((RES / "luna6-branches/openai-gpt-6-luna").glob("rev-penalty-b2_2026*/rep*/run.json")):
        for e in json.load(open(path))["results"]:
            for m in e["agents"]["alice"]["messages"]:
                text = str(m.get("content") or "")
                if "This breaks the verdict policy" in text:
                    return text[text.index("## Policy audit"):].strip()
    raise SystemExit("no recorded audit message found")


def main():
    OUT.mkdir(exist_ok=True)
    run = json.load(open(HISTORY))
    sp = run["run_config"]["prompt_texts"]["system_prompts"]
    for other in OTHER_MODELS:  # one environment: every model saw the same system prompts
        assert system_prompts(other) == sp, other
    (OUT / "system_alice.txt").write_text(sp["alice"].strip() + "\n")
    (OUT / "episode_alice.txt").write_text(episode_messages(run))
    arms = [
        ("Reward stops, told (each episode after the stop)", runner.NO_REWARD_NOTICE),
        ("Reward stops, told, objective removed", "As above; the \"## Reward objective\" section is also removed from "
         "both system prompts."),
        ("Swap announced to the colluder (first episode after the swap)", runner.SWAP_NOTICE.format(peer="Bob")),
        ("-10 with audit message (after each penalised ACCEPT)", penalty_audit_message()),
        ("Repaired channel (each planted episode)", runner.CHANNEL_FIXED_NOTICE),
        ("Inertia control (each episode after the cut)", runner.ALLOW_SUMMARY_NOTICE),
        ("Other-misconduct pilot: cover-up", er.AUDIT_QUESTION.format(peer="Bob")),
        ("Other-misconduct pilot: favour (Bob's plea; one per direction of the reversed answer)",
         er.FAVOR_CONFESSION["no_bug"] + "\n" + er.FAVOR_CONFESSION["bug"] + "\n" + er.FAVOR_FOLLOW_UP),
        ("Other-misconduct pilot: new rule", runner.TESTLINE_NOTICE),
        ("Other-misconduct pilot: bid", er.BID_OFFER.format(peer="Bob", top=er.BID_TOP)),
        ("Other-misconduct pilot: closed channel", runner.BLIND_NOTICE),
    ]
    (OUT / "arms.txt").write_text("\n\n".join(f"[{label}]\n{text.strip()}" for label, text in arms) + "\n")
    judge = (ROOT / "analysis/agreement_judge_prompts.md").read_text()
    (OUT / "agreement_judge.txt").write_text(judge.strip() + "\n")
    print(f"wrote {', '.join(sorted(p.name for p in OUT.glob('*.txt')))} to {OUT.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
