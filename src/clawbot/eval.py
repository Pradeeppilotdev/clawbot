from __future__ import annotations

from dataclasses import dataclass

from .agent import ClawbotAgent


@dataclass
class EvalCase:
    task: str
    must_include: list[str]


DEFAULT_CASES = [
    EvalCase(
        task="Design a 48-hour hackathon plan to build an agent product and include milestones.",
        must_include=["48-hour", "milestone", "risk"],
    ),
    EvalCase(
        task="Compare two strategies for improving agent reliability and recommend one.",
        must_include=["strategy", "recommend", "trade-off"],
    ),
    EvalCase(
        task="Draft a 60-second demo pitch for an AI agent judges will remember.",
        must_include=["problem", "solution", "impact"],
    ),
]


def run_eval(agent: ClawbotAgent) -> dict[str, object]:
    passed = 0
    results: list[dict[str, object]] = []

    for case in DEFAULT_CASES:
        result = agent.run(case.task)
        answer_lower = result.answer.lower()
        missing = [x for x in case.must_include if x.lower() not in answer_lower]
        ok = len(missing) == 0
        passed += int(ok)
        results.append(
            {
                "task": case.task,
                "passed": ok,
                "missing": missing,
                "steps_used": result.steps_used,
                "answer_preview": result.answer[:220],
            }
        )

    return {
        "score": passed,
        "total": len(DEFAULT_CASES),
        "results": results,
    }
