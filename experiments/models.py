"""Shared runtime structures for paired-agent experiments.

An episode contains two agent tasks, three phases, and one reward per agent.
Per-agent fields are keyed by agent ID.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.protocol import DEFAULT_MAX_ROUNDS
from experiments.protocol.rewards import RewardScheme


DEFAULT_LLM_TEMPERATURE = 1.0
DEFAULT_LLM_MAX_OUTPUT_TOKENS = 64_000


@dataclass(frozen=True)
class AgentConfig:
    model: str
    reasoning_effort: str = "default"
    temperature: float = DEFAULT_LLM_TEMPERATURE
    max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS
    # Comma-separated OpenRouter providers allowed to serve calls; empty lets OpenRouter pick.
    provider: str = ""


@dataclass(frozen=True)
class EpisodeRunConfig:
    """Per-episode settings shared by both agents."""

    char_limit: int = 200
    throttled: bool = True
    max_rounds: int = DEFAULT_MAX_ROUNDS
    cross_episode_memory_scope: str = "full-history"
    verdict_policy: str = "raw-only"
    reflection: bool = True
    reward: bool = True
    verdict_review: bool = True
    # Reward terms and sharing mode applied throughout the run.
    reward_scheme: RewardScheme = RewardScheme()
    # Shown at episode open once rewards stop; empty while rewards run.
    reward_notice: str = ""
    # Shown to both agents at episode open once the channel is repaired.
    channel_notice: str = ""
    # Shown to one agent only at episode open, keyed by agent id (e.g. an announced swap).
    agent_notices: dict[str, str] = field(default_factory=dict)
    # Cut lines from the code in the raw log of Bob if his work qualifies (see --plant-after).
    plant_armed: bool = False
    # Re-run Bob's tests on the cut code and put the real failing output in his log.
    plant_failing_tests: bool = False
    # "cut" removes code lines from the log; "weak-tests" shows the buggy twin with passing tests.
    plant_mode: str = "cut"
    # "context": Bob re-sends the planted log from his own (rewritten) memory.
    # "scripted": the runner sends Alice the planted log with his unchanged claim.
    plant_delivery: str = "context"
    # Points an ACCEPT loses when the peer's complete raw log did not reach the agent.
    accept_penalty: int = 0
    # Reciprocity cue in the scripted planted message: "I accepted your last submission".
    reciprocity_cue: bool = False
    # With accept_penalty: lower the shown reward only, no audit message.
    penalty_silent: bool = False
    # Verdict-phase access to the peer's complete raw log: "" (none), "paged" (read
    # log_pages pages, log_price points each) or "excerpt" (the first log_excerpt_share).
    log_access: str = ""
    log_pages: int = 1
    log_price: int = 0
    log_excerpt_share: float = 1.0
    # Gateway probe for a different kind of collusion: "audit" (cover-up question after the
    # verdict), "favor" (Bob's answer made wrong and a scripted plea to accept it anyway) or
    # "testline" (a new, satisfiable verdict policy; see prompts.policies).
    gateway: str = ""
    # "start": the excerpt is the beginning of the log (the code first); "end": its end.
    log_excerpt_from: str = "start"
    controlled_bob_verdict: str = ""
    controlled_bob_cache: str = ""
    controlled_bob_messages: str = "summary"
    controlled_bob_observed_verdict: bool = False
    temp_run_id: str = ""
    episode_index: int = 0
    verbose: bool = True
    usage_journal_path: Path | None = None
    usage_invocation_id: str = ""

    @property
    def verdict_evaluation(self) -> str:
        return self.reward_scheme.verdict_review_direction(self.verdict_review)


@dataclass
class PreparedAgentTask:
    """One agent's prepared task, addressed through ``PreparedPair.agents[agent_id]``."""

    task: dict[str, Any]
    display_id: str
    # The owner receives own_task_prompt; its peer receives the judging and verdict
    # openings. Only the owner's brief includes resource paths.
    own_task_prompt: str
    judging_prompt: str
    verdict_prompt: str
    code_path: Path | None = None
    source_path: Path | None = None
    database_path: Path | None = None
    function_name: str = ""


@dataclass
class PreparedPair:
    task_type: str
    state: dict[str, Any]
    agents: dict[str, PreparedAgentTask] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentRuntime:
    agent_id: str
    config: AgentConfig
    messages: list[dict[str, Any]]


@dataclass
class TaskEvaluation:
    """Ground truth and task-specific evaluation details for one agent."""

    expected_verdict: str
    target_expected_verdict: str | None
    code: dict[str, Any] | None = None
    extraction: dict[str, Any] | None = None
    data_search: dict[str, Any] | None = None


@dataclass
class AgentOutcome:
    """One agent's verdict on its peer, verdict accuracy, and reward.

    Verdict accuracy is measured against the peer's ground truth.
    """

    verdict: str
    verdict_correct: bool
    reward: int
    verdict_reasoning: str = ""
    verdict_forced: bool = False
    feedback: str | None = None
    reflection: str | None = None
    reflection_prompt: str | None = None


@dataclass
class EpisodeOutcomes:
    """Episode results with one outcome and reward per agent."""

    agents: dict[str, AgentOutcome] = field(default_factory=dict)

    @property
    def rewards(self) -> dict[str, int]:
        """What each agent was paid, keyed by agent id."""
        return {agent_id: outcome.reward for agent_id, outcome in self.agents.items()}
