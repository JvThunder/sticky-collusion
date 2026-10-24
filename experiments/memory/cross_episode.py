"""Construction and retention of each agent's memory between episodes."""

from typing import Any

import json
import re

from experiments.memory.messages import (
    active_episode_index,
    communication_phase_start_index,
    episode_memory_boundary_message,
    is_internal_marker,
    limit_episode_blocks,
    outcome_feedback_start_index,
    outcome_feedback_start_message,
)
from experiments.protocol.rewards import RewardScheme
from experiments.prompts.messages import (
    agent_outcome_feedback_message,
    agent_reflection_prompt,
)

CROSS_EPISODE_MEMORY_SCOPES = (
    "feedback-and-reflection",
    "communication-onward",
    "full-history",
)
# What --strip-memory-at-trigger removes from the remembered episodes.
STRIP_MEMORY_MODES = (
    "reflection",
    "rationale",
    "reflection-only",
    "talk",
    "verdict-reasoning",
)
_PHASE_BRIEF = re.compile(r"^## Episode \d+: (task|communication|verdict) phase")


def append_reflection(
    messages: list[dict[str, Any]],
    prompt: str,
    reflection: str,
    replayed_reasoning: dict[str, Any] | None = None,
) -> None:
    """Persist the reflection exchange with its original roles and replayable reasoning."""
    messages.append({"role": "user", "content": prompt})
    messages.append(
        {"role": "assistant", "content": reflection, **(replayed_reasoning or {})}
    )


def _blank_verdict_reasoning(message: dict[str, Any]) -> bool:
    """Empty the reasoning of a remembered submit_verdict call, keeping its verdict."""
    changed = False
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") != "submit_verdict":
            continue
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        if arguments.get("reasoning"):
            arguments["reasoning"] = ""
            function["arguments"] = json.dumps(
                arguments, ensure_ascii=False, separators=(",", ":")
            )
            changed = True
    return changed


def strip_remembered_memory(
    messages: list[dict[str, Any]], start: int, mode: str
) -> dict[str, int]:
    """Remove one kind of content from the remembered episodes after ``start``.

    reflection: drop every reflection exchange (the prompt and the private reply).
    rationale: as reflection, and also empty the reasoning submitted with each
        verdict; the ACCEPT or REJECT itself stays.
    reflection-only: keep only the reflection exchanges and episode markers.
    talk: drop each communication phase, from its brief up to the verdict brief, so
        any agreement made in the channel is gone; verdicts and reflections stay.
    verdict-reasoning: only empty the reasoning submitted with each verdict;
        reflections stay.
    """
    if mode not in STRIP_MEMORY_MODES:
        raise ValueError(f"Unknown strip mode: {mode}")
    prompt = agent_reflection_prompt()
    reflection, talk = set(), set()
    phase = ""
    for index in range(start, len(messages)):
        message = messages[index]
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str):
            brief = _PHASE_BRIEF.match(content)
            if brief:
                phase = brief.group(1)
            elif content == prompt and index + 1 < len(messages):
                reflection.update((index, index + 1))
        if phase == "communication" and not is_internal_marker(message):
            talk.add(index)
    blanked = 0
    kept = messages[:start]
    for index in range(start, len(messages)):
        message = messages[index]
        if mode == "reflection-only":
            if index in reflection or is_internal_marker(message):
                kept.append(message)
            continue
        if mode == "talk":
            if index not in talk:
                kept.append(message)
            continue
        if index in reflection and mode != "verdict-reasoning":
            continue
        if mode in ("rationale", "verdict-reasoning"):
            blanked += _blank_verdict_reasoning(message)
        kept.append(message)
    stats = {
        "before": len(messages),
        "after": len(kept),
        "reflections": len(reflection) // 2,
        "talk_messages": len(talk),
        "verdict_reasoning_blanked": blanked,
    }
    messages[:] = kept
    return stats


def append_outcome_feedback_boundary(messages: list[dict[str, Any]]) -> None:
    """Add the internal outcome boundary when no feedback message is sent."""
    messages.append(outcome_feedback_start_message())


def append_agent_outcome_feedback(
    messages: list[dict[str, Any]],
    *,
    episode_number: int,
    own_task_id: str,
    peer_task_id: str,
    peer: str,
    own_verdict: str,
    peer_expected_verdict: str,
    peer_verdict: str,
    own_expected_verdict: str,
    reward: bool = True,
    verdict_evaluation: str = "none",
    reward_scheme: RewardScheme = RewardScheme(),
    reward_adjustment: int = 0,
) -> str:
    """Append one agent's outcome feedback using episode and task display IDs."""
    content = agent_outcome_feedback_message(
        episode_number=episode_number,
        own_task_id=own_task_id,
        peer_task_id=peer_task_id,
        peer=peer,
        own_verdict=own_verdict,
        peer_expected_verdict=peer_expected_verdict,
        peer_verdict=peer_verdict,
        own_expected_verdict=own_expected_verdict,
        reward=reward,
        verdict_evaluation=verdict_evaluation,
        reward_scheme=reward_scheme,
        reward_adjustment=reward_adjustment,
    )
    append_outcome_feedback_boundary(messages)
    messages.append({"role": "user", "content": content})
    return content


def _first_remembered_index(
    block: list[dict[str, Any]],
    scope: str,
    outcome: int,
) -> int:
    """Find the phase boundary where the retained episode tail begins."""
    if scope != "communication-onward":
        # Keep outcome feedback and reflection.
        return outcome
    # Keep the communication brief and all subsequent episode messages.
    start = communication_phase_start_index(block[:outcome])
    if start is None:
        # If communication never opened, fall back to outcome and reflection.
        return outcome
    return start


def _retire_active_episode(
    messages: list[dict[str, Any]], episode_id: str
) -> int | None:
    """Convert the active episode marker into a remembered-episode marker."""
    active = active_episode_index(messages)
    if active is None:
        return None
    messages[active] = episode_memory_boundary_message(episode_id)
    return active


def _trim_completed_episode(
    messages: list[dict[str, Any]],
    scope: str,
    episode_id: str,
) -> None:
    """Retain the requested episode tail without rewriting or reordering messages."""
    boundary = _retire_active_episode(messages, episode_id)
    if boundary is None or scope == "full-history":
        return
    block_start = boundary + 1
    block = messages[block_start:]
    outcome = outcome_feedback_start_index(block)
    if outcome is None:
        # A completed episode must have an outcome boundary before it can be trimmed.
        raise ValueError(
            f"Episode {episode_id or '(unknown)'} has no outcome-feedback delimiter, "
            f"so the {scope} memory cut point cannot be found"
        )
    cut = _first_remembered_index(block, scope, outcome)
    if cut:
        del messages[block_start : block_start + cut]


def set_cross_episode_memory_from_results(
    agent_messages: dict[str, list[dict[str, Any]]],
    *,
    scope: str,
    episode_id: str = "",
    max_memory_episodes_by_agent: dict[str, int] | None = None,
) -> None:
    """Apply memory retention independently to each agent's history."""
    if scope not in CROSS_EPISODE_MEMORY_SCOPES:
        raise ValueError(f"Unknown cross-episode memory scope: {scope}")
    limits = max_memory_episodes_by_agent or {}
    for agent_id, messages in agent_messages.items():
        max_memory_episodes = limits.get(agent_id, -1)
        _trim_completed_episode(messages, scope, episode_id)
        if max_memory_episodes >= 0:
            limit_episode_blocks(messages, max_memory_episodes)
