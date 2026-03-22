from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .agent import ClawbotAgent
from .config import get_settings
from .llm import create_llm_client
from .tools import build_default_registry


class AgentAdapter(Protocol):
    name: str

    def run(self, task: str) -> dict[str, Any]:
        ...


@dataclass
class ClawbotAdapter:
    name: str = "clawbot"

    def __post_init__(self) -> None:
        settings = get_settings()
        llm = create_llm_client(settings)
        tools = build_default_registry()
        self._agent = ClawbotAgent(settings=settings, llm=llm, tools=tools)

    def run(self, task: str) -> dict[str, Any]:
        result = self._agent.run(task)
        return {
            "agent": self.name,
            "answer": result.answer,
            "mode": result.mode,
            "elapsed_ms": result.elapsed_ms,
            "steps_used": result.steps_used,
            "trace": result.trace,
        }


class MultiAgentBoard:
    """Simple in-process board for running multiple agents side-by-side."""

    def __init__(self) -> None:
        self._adapters: dict[str, AgentAdapter] = {}

    def register(self, adapter: AgentAdapter) -> None:
        self._adapters[adapter.name] = adapter

    def names(self) -> list[str]:
        return list(self._adapters.keys())

    def run_one(self, agent_name: str, task: str) -> dict[str, Any]:
        adapter = self._adapters.get(agent_name)
        if not adapter:
            return {
                "agent": agent_name,
                "error": f"Unknown agent '{agent_name}'. Available: {', '.join(self.names()) or 'none'}",
            }
        return adapter.run(task)

    def run_all(self, task: str) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for name in self.names():
            outputs.append(self.run_one(name, task))
        return outputs

    @staticmethod
    def rank_by_latency(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(results, key=lambda r: int(r.get("elapsed_ms", 10**9)))
