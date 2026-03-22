from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agent import AgentResult


@dataclass
class Scorecard:
    task_success: int
    latency_score: int
    recovery_score: int
    overall_score: int
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_success": self.task_success,
            "latency_score": self.latency_score,
            "recovery_score": self.recovery_score,
            "overall_score": self.overall_score,
            "details": self.details,
        }


def build_scorecard(task: str, result: AgentResult) -> Scorecard:
    answer = result.answer.strip()
    has_actionable_shape = _is_actionable(answer)
    task_success = 100 if has_actionable_shape else 55

    latency_score = _latency_score(result.elapsed_ms)

    recovery_score = _recovery_score(result)

    overall = int(round(task_success * 0.5 + latency_score * 0.3 + recovery_score * 0.2))

    details = {
        "task": task,
        "mode": result.mode,
        "elapsed_ms": result.elapsed_ms,
        "steps_used": result.steps_used,
        "events_count": len(result.trace),
        "has_sources": "Sources" in answer,
        "has_numbered_plan": bool(_count_numbered_lines(answer) >= 3),
    }

    return Scorecard(
        task_success=task_success,
        latency_score=latency_score,
        recovery_score=recovery_score,
        overall_score=overall,
        details=details,
    )


def _latency_score(elapsed_ms: int) -> int:
    if elapsed_ms <= 3000:
        return 100
    if elapsed_ms <= 6000:
        return 92
    if elapsed_ms <= 10000:
        return 80
    if elapsed_ms <= 20000:
        return 65
    return 45


def _recovery_score(result: AgentResult) -> int:
    trace_text = "\n".join(str(x) for x in result.trace).lower()

    provider_outage = "temporarily unavailable" in result.answer.lower() or "llm_provider_error" in trace_text
    deterministic_fallback = "deterministic fallback" in result.answer.lower()

    if not provider_outage:
        return 90
    if deterministic_fallback and len(result.answer) > 220:
        return 85
    if deterministic_fallback:
        return 70
    return 40


def _is_actionable(answer: str) -> bool:
    lower = answer.lower()
    bullets = answer.count("\n-") + answer.count("\n1.")
    has_plan_words = any(x in lower for x in ["plan", "steps", "action", "metrics", "risk"])
    return len(answer) >= 240 and (bullets >= 3 or has_plan_words)


def _count_numbered_lines(answer: str) -> int:
    count = 0
    for line in answer.splitlines():
        stripped = line.strip()
        if len(stripped) >= 2 and stripped[0].isdigit() and stripped[1] == ".":
            count += 1
    return count
